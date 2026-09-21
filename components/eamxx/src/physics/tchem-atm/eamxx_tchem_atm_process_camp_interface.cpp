#include "eamxx_tchem_atm_process_camp_interface.hpp"
#include "eamxx_tchem_atm_tchem_functions.hpp"

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>

#include "share/physics/eamxx_common_physics_functions.hpp"

#include <algorithm>

namespace scream {

// ============================================================================
// Constructor
// ============================================================================
TChemATMCamp::TChemATMCamp(const ekat::Comm& comm,
                           const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

// ============================================================================
// create_requests — register fields and build TChem metadata
// ============================================================================
void TChemATMCamp::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;
  using namespace ShortFieldTagsNames;

  m_grid = m_grids_manager->get_grid("physics");
  EKAT_REQUIRE_MSG(m_grid != nullptr,
                   "Error! TChemATMCamp could not get 'physics' grid.\n");

  const auto chem_file = m_params.get<std::string>(
      "chem_file", m_params.get<std::string>("chemfile", ""));
  EKAT_REQUIRE_MSG(!chem_file.empty(),
                   "Error! Missing required parameter 'chem_file' for "
                   "tchem_atm_camp.\n");

  const auto aero_file = m_params.get<std::string>(
      "aero_file", m_params.get<std::string>("aerofile", ""));
  EKAT_REQUIRE_MSG(!aero_file.empty(),
                   "Error! Missing required parameter 'aero_file' for "
                   "tchem_atm_camp.\n");

  const auto& grid_name   = m_grid->name();
  const auto  scalar3d_mid = m_grid->get_3d_scalar_layout(LEV);

  // Minimal field requirements — only T, P, and qv for gas-phase chemistry.
  add_field<Required>("p_mid", scalar3d_mid, Pa, grid_name);
  add_field<Required>("T_mid", scalar3d_mid, K, grid_name);
  add_field<Required>("qv", scalar3d_mid, q_unit, grid_name);

  // ----------------------------------------------------------------
  // Build gas-phase kinetic model
  // ----------------------------------------------------------------
  if (m_atm_logger) m_atm_logger->debug("[TChemATMCamp] KineticModelData");
  m_kmd  = TChem::KineticModelData(chem_file);
  m_kmcd = TChem::createNCAR_KineticModelConstData<tchem_device_type>(m_kmd);
  if (m_atm_logger)
    m_atm_logger->debug("[TChemATMCamp] nSpec = " +
                        std::to_string(m_kmd.nSpec_));

  // ----------------------------------------------------------------
  // Build aerosol model (needed by AerosolChemistry_Problem)
  // ----------------------------------------------------------------
  if (m_atm_logger) m_atm_logger->debug("[TChemATMCamp] AerosolModelData");
  m_amd  = TChem::AerosolModelData(aero_file, m_kmd);
  m_amcd = TChem::create_AerosolModelConstData<tchem_device_type>(m_amd);

  m_num_const_spec = m_kmcd.nConstSpec;
  m_n_active_gas_vars = m_kmcd.nSpec - m_num_const_spec;

  // Look up constant-species indices by name from the mechanism.
  // These map to positions in the full species array (species_indx_)
  // which determines the state-vector column (index + 3).
  {
    const auto& idx = m_kmd.species_indx_;
    auto find_species = [&](const std::string& name) -> int {
      auto it = idx.find(name);
      EKAT_REQUIRE_MSG(it != idx.end(),
                       "Error! Constant species '" + name +
                           "' not found in mechanism.\n");
      return it->second;
    };
    m_idx_M   = find_species("M");
    m_idx_O2  = find_species("O2");
    m_idx_N2  = find_species("N2");
    m_idx_H2O = find_species("H2O");
    m_idx_H2  = find_species("H2");
    m_idx_CH4 = find_species("CH4");
  }

  m_total_n_species = m_kmcd.nSpec + m_amcd.nSpec * m_amcd.nParticles;
  m_n_active_vars   = m_n_active_gas_vars + m_amcd.nSpec * m_amcd.nParticles;
  m_state_vec_dim   = TChem::Impl::getStateVectorSize(m_total_n_species);

  if (m_atm_logger) {
    m_atm_logger->info("[TChemATMCamp] nSpec (gas)       = " +
                       std::to_string(m_kmcd.nSpec));
    m_atm_logger->info("[TChemATMCamp] nConstSpec        = " +
                       std::to_string(m_num_const_spec));
    m_atm_logger->info("[TChemATMCamp] n_active_gas_vars = " +
                       std::to_string(m_n_active_gas_vars));
    m_atm_logger->info("[TChemATMCamp] nAeroSpec         = " +
                       std::to_string(m_amcd.nSpec));
    m_atm_logger->info("[TChemATMCamp] nParticles        = " +
                       std::to_string(m_amcd.nParticles));
    m_atm_logger->info("[TChemATMCamp] state_vec_dim     = " +
                       std::to_string(m_state_vec_dim));
  }

  // ----------------------------------------------------------------
  // Molecular weights (only for active gas species)
  // ----------------------------------------------------------------
  m_species_mw.resize(m_kmd.nSpec_, 1.0);
  EKAT_REQUIRE_MSG(
      m_params.isSublist("molecular_weights"),
      "Error! Missing required sublist 'molecular_weights' under "
      "tchem_atm_camp parameters.\n");
  const auto& mw_list = m_params.sublist("molecular_weights");
  m_species_names_host = m_kmd.sNames_.view_host();

  for (int i = 0; i < m_n_active_gas_vars; ++i) {
    const std::string sname(&m_species_names_host(i, 0));
    EKAT_REQUIRE_MSG(mw_list.isParameter(sname),
                     "Error! Molecular weight not found for species '" +
                         sname +
                         "' in 'molecular_weights' sublist.\n");
    m_species_mw[i] = mw_list.get<double>(sname);
  }
  if (m_atm_logger)
    m_atm_logger->debug("[TChemATMCamp] Done loading molecular weights");

  m_tchem_ready = true;

  // ----------------------------------------------------------------
  // Register tracer fields for active gas species
  // ----------------------------------------------------------------
  for (int i = 0; i < m_n_active_gas_vars; ++i) {
    const std::string sname(&m_species_names_host(i, 0));
    add_tracer<Updated>(sname, m_grid, q_unit);
  }
  if (m_atm_logger)
    m_atm_logger->info("[TChemATMCamp] Tracers registered: " +
                       std::to_string(m_n_active_gas_vars));
}

// ============================================================================
// initialize_impl — allocate views, set up solvers
// ============================================================================
void TChemATMCamp::initialize_impl(const RunType /* run_type */) {
  EKAT_REQUIRE_MSG(
      m_tchem_ready,
      "Error! TChemATMCamp::initialize_impl called before TChem model "
      "initialization.\n");

  m_ncols  = m_grid->get_num_local_dofs();
  m_nlevs  = m_grid->get_num_vertical_levels();
  m_nbatch = m_ncols * m_nlevs;
  if (m_nbatch == 0) return;

  // Allocate TChem state array and working views
  m_state =
      real_type_2d_view("camp_state", m_nbatch, m_state_vec_dim);
  m_num_concentration =
      real_type_2d_view("camp_num_conc", m_nbatch, m_amd.nParticles_);
  m_t       = real_type_1d_view("camp_time", m_nbatch);
  m_dt_view = real_type_1d_view("camp_dt", m_nbatch);
  m_tadv    = TChem::time_advance_type_1d_view("camp_tadv", m_nbatch);

  // Sampling index views (all levels — chemistry_domain = "all")
  m_sample_icol = view_1d_int("camp_sample_icol", m_nbatch);
  m_sample_ilev = view_1d_int("camp_sample_ilev", m_nbatch);
  {
    const int nlev = m_nlevs;
    const auto sample_icol = m_sample_icol;
    const auto sample_ilev = m_sample_ilev;
    Kokkos::parallel_for(
        "camp_fill_all_indices",
        Kokkos::RangePolicy<TChem::exec_space>(0, m_nbatch),
        KOKKOS_LAMBDA(const int isample) {
          sample_icol(isample) = isample / nlev;
          sample_ilev(isample) = isample % nlev;
        });
  }

  // ------- Read solver parameters from namelist -------
  m_solver_type =
      m_params.get<std::string>("solver_type", "tines");
  if (m_solver_type == "tines") {
    m_solver_enum = SolverType::Tines;
  } else if (m_solver_type == "cvode_batch") {
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
    m_solver_enum = SolverType::CVODEBatch;
#else
    EKAT_REQUIRE_MSG(
        false,
        "Error! solver_type 'cvode_batch' requires "
        "TCHEM_ATM_ENABLE_SUNDIALS=ON.\n");
#endif
  } else {
    EKAT_REQUIRE_MSG(false,
                     "Error! Unknown solver_type '" + m_solver_type +
                         "'. Valid options: tines, cvode_batch.\n");
  }
  if (m_atm_logger)
    m_atm_logger->info("[TChemATMCamp] solver_type = " + m_solver_type);

  // Tines implicit-euler parameters
  if (m_params.isSublist("implicit_euler_parameters")) {
    auto& ie = m_params.sublist("implicit_euler_parameters");
    m_max_time_iterations   = ie.get<int>("max_time_iterations", 1000);
    m_max_newton_iterations = ie.get<int>("max_newton_iterations", 100);
    m_jacobian_interval     = ie.get<int>("jacobian_interval", 1);
    m_dtmin_sub   = ie.get<double>("dtmin_sub", 1e-4);
    m_dtmax_sub   = ie.get<double>("dtmax_sub", -1.0);
    m_atol_newton = ie.get<double>("atol_newton", 1e-10);
    m_rtol_newton = ie.get<double>("rtol_newton", 1e-6);
    m_atol_time   = ie.get<double>("atol_time", 1e-12);
    m_rtol_time   = ie.get<double>("rtol_time", 1e-4);
  }

  // CVODE parameters
  if (m_params.isSublist("cvode_parameters")) {
    auto& cv = m_params.sublist("cvode_parameters");
    m_cvode_rtol      = cv.get<double>("rtol", 1e-8);
    m_cvode_atol      = cv.get<double>("atol", 1e-12);
    m_cvode_max_steps = cv.get<int>("max_steps", 10000);
    m_cvode_max_step  = cv.get<double>("max_step", -1.0);
    m_cvode_min_step  = cv.get<double>("min_step", 0.0);
  }

  // ------- Allocate tolerance views for Tines solver -------
  if (m_solver_enum == SolverType::Tines) {
    const TChem::ordinal_type n_eq =
        problem_type::getNumberOfTimeODEs(m_kmcd, m_amcd);

    m_tol_newton = real_type_1d_view("camp_tol_newton", 2);
    m_tol_time   = real_type_2d_view("camp_tol_time", n_eq, 2);
    m_fac        = real_type_2d_view("camp_fac", m_nbatch, n_eq);

    auto tol_newton_h = Kokkos::create_mirror_view(m_tol_newton);
    auto tol_time_h   = Kokkos::create_mirror_view(m_tol_time);
    tol_newton_h(0) = m_atol_newton;
    tol_newton_h(1) = m_rtol_newton;
    for (TChem::ordinal_type i = 0; i < n_eq; ++i) {
      tol_time_h(i, 0) = m_atol_time;
      tol_time_h(i, 1) = m_rtol_time;
    }
    Kokkos::deep_copy(m_tol_newton, tol_newton_h);
    Kokkos::deep_copy(m_tol_time, tol_time_h);
    Kokkos::deep_copy(m_fac, 0.0);
  }

  // ------- CVODE batch solver initialization -------
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_solver_enum == SolverType::CVODEBatch) {
    const TChem::ordinal_type n_eq =
        problem_type::getNumberOfTimeODEs(m_kmcd, m_amcd);

    if (m_atm_logger)
      m_atm_logger->info("[TChemATMCamp] Initializing CVODE batch solver");

    m_sundials_ctx = std::make_unique<sundials::Context>();

    CVODESizeType length{
        static_cast<CVODESizeType>(m_nbatch * n_eq)};
    m_cvode_y      = std::make_unique<CVODEVecType>(length, *m_sundials_ctx);
    m_cvode_abstol = std::make_unique<CVODEVecType>(length, *m_sundials_ctx);
    N_VConst(SUN_RCONST(m_cvode_atol), *m_cvode_abstol);

    m_cvode_mem = CVodeCreate(CV_BDF, *m_sundials_ctx);
    EKAT_REQUIRE_MSG(m_cvode_mem != nullptr,
                     "Error! CVodeCreate failed.\n");

    // Use TChem's built-in AerosolChemistry_CVODE_K RHS callback
    int rv = CVodeInit(m_cvode_mem, TChem::AerosolChemistry_CVODE_K::f,
                       SUN_RCONST(0.0), *m_cvode_y);
    EKAT_REQUIRE_MSG(rv >= 0,
                     "Error! CVodeInit failed with code " +
                         std::to_string(rv) + ".\n");

    rv = CVodeSVtolerances(m_cvode_mem, SUN_RCONST(m_cvode_rtol),
                           *m_cvode_abstol);
    EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSVtolerances failed.\n");

    rv = CVodeSetMaxNumSteps(m_cvode_mem, m_cvode_max_steps);
    EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSetMaxNumSteps failed.\n");

    if (m_cvode_max_step > 0.0) {
      rv = CVodeSetMaxStep(m_cvode_mem, SUN_RCONST(m_cvode_max_step));
      EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSetMaxStep failed.\n");
    }
    if (m_cvode_min_step > 0.0) {
      rv = CVodeSetMinStep(m_cvode_mem, SUN_RCONST(m_cvode_min_step));
      EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSetMinStep failed.\n");
    }

    // Allocate CVODE user data views
    m_cvode_temperature    = real_type_1d_view("cvode_camp_T", m_nbatch);
    m_cvode_pressure       = real_type_1d_view("cvode_camp_P", m_nbatch);
    m_cvode_const_tracers  = real_type_2d_view("cvode_camp_const",
                                               m_nbatch, m_num_const_spec);

    // Populate the TChem UserData struct (defined in
    // TChem_AerosolChemistry_CVODE_RHS_Jacobian.hpp)
    m_cvode_udata.nbatches          = m_nbatch;
    m_cvode_udata.batchSize         = n_eq;
    m_cvode_udata.kmcd              = m_kmcd;
    m_cvode_udata.amcd              = m_amcd;
    m_cvode_udata.temperature       = m_cvode_temperature;
    m_cvode_udata.pressure          = m_cvode_pressure;
    m_cvode_udata.const_tracers     = m_cvode_const_tracers;
    m_cvode_udata.num_concentration = m_num_concentration;
    m_cvode_udata.fac =
        real_type_2d_view("cvode_camp_fac", m_nbatch, n_eq);
    Kokkos::deep_copy(m_cvode_udata.fac, 0.0);

#if defined(TCHEM_ATM_ENABLE_GPU)
    m_cvode_udata.JacRL =
        real_type_3d_view("cvode_camp_JacRL", m_nbatch, n_eq, n_eq);
#endif

    rv = CVodeSetUserData(m_cvode_mem, &m_cvode_udata);
    EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSetUserData failed.\n");

    // Use Kokkos dense block diagonal matrix with explicit Jacobian
    // (solver_type = 0 in the TChem AerosolChemistry_CVODE_K example).
    m_cvode_A = std::make_unique<CVODEMatType>(
        m_nbatch, n_eq, n_eq, *m_sundials_ctx);
    m_cvode_LS = std::make_unique<CVODELSType>(*m_sundials_ctx);

    rv = CVodeSetLinearSolver(m_cvode_mem, m_cvode_LS->Convert(),
                              m_cvode_A->Convert());
    EKAT_REQUIRE_MSG(rv >= 0,
                     "Error! CVodeSetLinearSolver (dense) failed.\n");

    rv = CVodeSetJacFn(m_cvode_mem,
                       TChem::AerosolChemistry_CVODE_K::Jac);
    EKAT_REQUIRE_MSG(rv >= 0,
                     "Error! CVodeSetJacFn failed.\n");

    // Set up team policy for CVODE callbacks
    const TChem::ordinal_type per_team_extent =
        problem_type::getWorkSpaceSize(m_kmcd, m_amcd) + n_eq;
    cvode_policy_type policy(TChem::exec_space(), m_nbatch,
                             Kokkos::AUTO());
    const TChem::ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));
    m_cvode_udata.policy = policy;

    if (m_atm_logger)
      m_atm_logger->info(
          "[TChemATMCamp] CVODE batch solver initialized successfully");
  }
#endif  // TCHEM_ATM_ENABLE_SUNDIALS
}

// ============================================================================
// run_impl — pack state, run gas-phase chemistry, unpack state
// ============================================================================
void TChemATMCamp::run_impl(const double dt) {
  EKAT_ASSERT_MSG(
      m_tchem_ready,
      "Error! TChemATMCamp::run_impl called before TChem model "
      "initialization.\n");

  using ordinal_type = TChem::ordinal_type;

  if (m_nbatch == 0) return;

  const auto& t_mid = get_field_in("T_mid").get_view<const Real**>();
  const auto& p_mid = get_field_in("p_mid").get_view<const Real**>();
  const auto& qv    = get_field_in("qv").get_view<const Real**>();
  const int nlevs    = m_nlevs;
  const int nsamples = m_nbatch;
  const auto state   = m_state;
  const auto sample_icol = m_sample_icol;
  const auto sample_ilev = m_sample_ilev;

  // ----------------------------------------------------------------
  // Initialize time-advance and solver views
  // ----------------------------------------------------------------
  Kokkos::deep_copy(m_t, 0.0);
  Kokkos::deep_copy(m_dt_view, dt);

  const Real dtmax_sub = (m_dtmax_sub > 0.0) ? m_dtmax_sub : dt;
  const Real dtmin_sub = m_dtmin_sub;
  TChem::time_advance_type tadv_default;
  tadv_default._tbeg = 0;
  tadv_default._tend = dt;
  tadv_default._dt   = dtmax_sub;
  tadv_default._dtmin = dtmin_sub;
  tadv_default._dtmax = dtmax_sub;
  tadv_default._max_num_newton_iterations  = m_max_newton_iterations;
  tadv_default._num_time_iterations_per_interval = 100;
  tadv_default._jacobian_interval = m_jacobian_interval;
  Kokkos::deep_copy(m_tadv, tadv_default);

  // ----------------------------------------------------------------
  // Pack pressure, temperature into TChem state vector
  // state layout: [density, pressure, temperature, species_0, species_1, …]
  // ----------------------------------------------------------------
  tchem::pack_into_state(state, p_mid, sample_icol, sample_ilev, nsamples,
                         1, "camp_pack_P");
  tchem::pack_into_state(state, t_mid, sample_icol, sample_ilev, nsamples,
                         2, "camp_pack_T");

  // Pack active gas species (wet-mmr → dry-vmr conversion)
  for (int ivar = 0; ivar < m_n_active_gas_vars; ++ivar) {
    const std::string sname(&m_species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(sname).get_view<Real**>();
    tchem::pack_wet_mmr_into_state(state, q_tracer, qv, sample_icol,
                                   sample_ilev, nsamples, ivar + 3,
                                   m_species_mw[ivar],
                                   "camp_pack_tracer");
  }

  // ----------------------------------------------------------------
  // Compute invariants (constant species)
  // M [molecules/cm^3] = Pa_xfac * P / (boltz_cgs * T)
  // ----------------------------------------------------------------
  constexpr Real Pa_xfac  = 10.0;
  constexpr Real boltz_cgs = 0.13806500000000001E-015;
  // Use the species indices looked up from the mechanism (+ 3 for the
  // density/pressure/temperature prefix in the state vector).
  const int col_M   = m_idx_M   + 3;
  const int col_O2  = m_idx_O2  + 3;
  const int col_N2  = m_idx_N2  + 3;
  const int col_H2O = m_idx_H2O + 3;
  const int col_H2  = m_idx_H2  + 3;
  const int col_CH4 = m_idx_CH4 + 3;

  Kokkos::parallel_for(
      "camp_compute_invariants",
      Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        const Real M_val =
            Pa_xfac * p_mid(icol, ilev) / (boltz_cgs * t_mid(icol, ilev));
        state(isample, col_M)   = M_val;
        state(isample, col_N2)  = 0.79;
        state(isample, col_O2)  = 0.21;
        state(isample, col_H2O) = qv(icol, ilev) / (1.0 + qv(icol, ilev));
        state(isample, col_H2)  = 5.5e-7;
        state(isample, col_CH4) = 0.0;
      });

  // Initialise aerosol concentrations in the state to zero
  // (aerosol species follow gas species in the state vector)
  const int aero_start = 3 + m_kmcd.nSpec;
  const int n_aero_total = m_amcd.nSpec * m_amcd.nParticles;
  if (n_aero_total > 0) {
    Kokkos::parallel_for(
        "camp_zero_aero",
        Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < n_aero_total; ++j) {
            state(isample, aero_start + j) = 0.0;
          }
        });
  }

  // Initialise num_concentration to zero (no particles)
  Kokkos::deep_copy(m_num_concentration, 0.0);

  // ================================================================
  // Time integration — Solver-specific paths
  // ================================================================

#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_solver_enum == SolverType::CVODEBatch) {
    // ------------------------------------------------------------------
    // CVODE Batch Solver Path (solver_type = 0 / dense from CVODE_K example)
    // ------------------------------------------------------------------
    const ordinal_type n_eq =
        problem_type::getNumberOfTimeODEs(m_kmcd, m_amcd);

    // Copy T, P, and constant tracers into CVODE user-data views
    const auto cvode_temperature   = m_cvode_temperature;
    const auto cvode_pressure      = m_cvode_pressure;
    const auto cvode_const_tracers = m_cvode_const_tracers;
    const auto nConstSpec          = m_num_const_spec;
    const auto n_active_gas        = m_n_active_gas_vars;

    Kokkos::parallel_for(
        "camp_cvode_copy_T_P",
        Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
        KOKKOS_LAMBDA(const int isample) {
          const int icol = sample_icol(isample);
          const int ilev = sample_ilev(isample);
          cvode_temperature(isample) = t_mid(icol, ilev);
          cvode_pressure(isample)    = p_mid(icol, ilev);
        });

    const int const_start = n_active_gas + 3;
    Kokkos::parallel_for(
        "camp_cvode_copy_const",
        Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < nConstSpec; ++j) {
            cvode_const_tracers(isample, j) =
                state(isample, const_start + j);
          }
        });

    // Pack active species into CVODE y vector:
    // First n_active_gas entries are the active gas species,
    // then the aerosol particle species.
    real_type_2d_view y2d(m_cvode_y->View().data(), nsamples, n_eq);
    const int nAeroTotal = n_aero_total;  // capture

    Kokkos::parallel_for(
        "camp_cvode_pack_y",
        Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
        KOKKOS_LAMBDA(const int isample) {
          // Active gas species
          for (int j = 0; j < n_active_gas; ++j) {
            y2d(isample, j) = state(isample, j + 3);
          }
          // Aerosol particle species
          for (int j = 0; j < nAeroTotal; ++j) {
            y2d(isample, n_active_gas + j) =
                state(isample, aero_start + j);
          }
        });

    m_cvode_udata.nbatches = nsamples;

    // Update policy for current sample count
    const TChem::ordinal_type rhs_extent =
        problem_type::getWorkSpaceSize(m_kmcd, m_amcd) + n_eq;
    cvode_policy_type policy(TChem::exec_space(), nsamples,
                             Kokkos::AUTO());
    const TChem::ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(rhs_extent);
    policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));
    m_cvode_udata.policy = policy;

    // Set max step to atmosphere dt if m_cvode_max_step < 0
    if (m_cvode_max_step < 0.0) {
      int rv = CVodeSetMaxStep(m_cvode_mem, SUN_RCONST(dt));
      EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeSetMaxStep failed.\n");
    }

    // Reinitialize CVODE for this timestep (t0=0, y0 = current y)
    int rv = CVodeReInit(m_cvode_mem, SUN_RCONST(0.0), *m_cvode_y);
    EKAT_REQUIRE_MSG(rv >= 0, "Error! CVodeReInit failed.\n");

    sunrealtype t_cvode = SUN_RCONST(0.0);
    rv = CVode(m_cvode_mem, SUN_RCONST(dt), *m_cvode_y, &t_cvode,
               CV_NORMAL);
    if (rv < 0 && m_atm_logger)
      m_atm_logger->warn("[TChemATMCamp] CVode returned error code " +
                         std::to_string(rv));

    // Unpack results back into state
    Kokkos::parallel_for(
        "camp_cvode_unpack_y",
        Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < n_active_gas; ++j) {
            state(isample, j + 3) = y2d(isample, j);
          }
        });

  } else
#endif  // TCHEM_ATM_ENABLE_SUNDIALS
  {
    // ------------------------------------------------------------------
    // Tines Implicit Solver Path (AerosolChemistry::runDeviceBatch)
    // ------------------------------------------------------------------
    using policy_type =
        typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

    policy_type policy(TChem::exec_space(), nsamples, Kokkos::AUTO());
    const ordinal_type per_team_extent =
        TChem::AerosolChemistry::getWorkSpaceSize(m_kmcd, m_amcd);
    const ordinal_type per_team_scratch =
        TChem::Scratch<real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));

    const auto t_view       = m_t;
    const auto dt_view_loc  = m_dt_view;
    const auto tadv         = m_tadv;

    TChem::TeamConfOutput team_conf_output;

    TChem::real_type tsum(0);
    for (int iter = 0;
         iter < m_max_time_iterations && tsum <= dt * 0.9999; ++iter) {

      TChem::AerosolChemistry::runDeviceBatch(
          policy, m_tol_newton, m_tol_time, m_fac, tadv, m_state,
          m_num_concentration, t_view, dt_view_loc, m_state,
          team_conf_output, m_kmcd, m_amcd);

      tsum = 0;
      Kokkos::parallel_reduce(
          "camp_update_tadv",
          Kokkos::RangePolicy<TChem::exec_space>(0, nsamples),
          KOKKOS_LAMBDA(const int i, TChem::real_type& update) {
            tadv(i)._tbeg = t_view(i);
            tadv(i)._dt   = dt_view_loc(i);
            update += t_view(i);
          },
          tsum);
      tsum /= nsamples;
    }
  }

  // ----------------------------------------------------------------
  // Unpack active gas species back to tracer fields
  // ----------------------------------------------------------------
  for (int ivar = 0; ivar < m_n_active_gas_vars; ++ivar) {
    const std::string sname(&m_species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(sname).get_view<Real**>();
    tchem::unpack_wet_mmr_from_state(q_tracer, state, qv, sample_icol,
                                     sample_ilev, nsamples, ivar + 3,
                                     m_species_mw[ivar],
                                     "camp_unpack_tracer");
  }
}

// ============================================================================
// finalize_impl — clean up CVODE resources
// ============================================================================
void TChemATMCamp::finalize_impl() {
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_cvode_mem != nullptr) {
    CVodeFree(&m_cvode_mem);
    m_cvode_mem = nullptr;
  }
  m_sundials_ctx.reset();
  m_cvode_y.reset();
  m_cvode_abstol.reset();
  m_cvode_A.reset();
  m_cvode_LS.reset();
#endif
}

}  // namespace scream
