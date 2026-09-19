#include "eamxx_tchem_atm_process_interface.hpp"
#include "eamxx_tchem_atm_tchem_functions.hpp"

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/mo_photo.hpp>

#include "physics/mam/readfiles/vertical_remapper_exo_coldens.hpp"
#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "share/algorithm/eamxx_data_interpolation.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"

#include <algorithm>

// CVODE batch solver support
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
#include "TChem_Impl_AtmosphericChemistryE3SM_Problem.hpp"
#include "TChem_Impl_NetProductionRates.hpp"
#endif

namespace scream {

// ============================================================================
// CVODE Batch Solver Static Callback Functions
// ============================================================================
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)

// RHS function: dy/dt = f(t,y)
// This is called by CVODE to evaluate the right-hand side for all batches
int TChemATM::cvode_rhs_func(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data) {
  auto* udata = static_cast<CVODEUserData*>(user_data);
  
  const auto nbatches = udata->nbatches;
  const auto batchSize = udata->batchSize;
  const auto policy = udata->policy;
  const auto kmcd = udata->kmcd;
  const auto temperature = udata->temperature;
  const auto pressure = udata->pressure;
  const auto const_tracers = udata->const_tracers;
  const auto photo_rates = udata->photo_rates;
  const auto external_sources = udata->external_sources;

  // Wrap NVector data as 2D Kokkos views
  cvode_real_type_2d_view vals(N_VGetDeviceArrayPointer(y), nbatches, batchSize);
  cvode_real_type_2d_view rhs(N_VGetDeviceArrayPointer(ydot), nbatches, batchSize);

  using problem_type = TChem::Impl::AtmosphericChemistryE3SM_Problem<Real, cvode_device_type>;
  
  const TChem::ordinal_type level = 1;
  const TChem::ordinal_type per_team_extent = problem_type::getWorkSpaceSize(kmcd);
  const std::string profile_name = "TChem::AtmosphericChemistryE3SM::CVODE_RHS";

  Kokkos::Profiling::pushRegion(profile_name);
  Kokkos::parallel_for(
      profile_name, policy,
      KOKKOS_LAMBDA(const typename cvode_policy_type::member_type& member) {
        const TChem::ordinal_type i = member.league_rank();
        
        const auto vals_at_i = Kokkos::subview(vals, i, Kokkos::ALL());
        const auto rhs_at_i = Kokkos::subview(rhs, i, Kokkos::ALL());
        const auto const_tracers_at_i = Kokkos::subview(const_tracers, i, Kokkos::ALL());
        
        cvode_real_type_1d_view photo_rates_at_i;
        if (photo_rates.extent(0) > 0) {
          photo_rates_at_i = Kokkos::subview(photo_rates, i, Kokkos::ALL());
        }
        
        cvode_real_type_1d_view external_sources_at_i;
        if (external_sources.extent(0) > 0) {
          external_sources_at_i = Kokkos::subview(external_sources, i, Kokkos::ALL());
        }
        
        // Get scratch memory and create a view from its raw pointer
        // This avoids the "incompatible spaces" error when passing to team_invoke
        TChem::Scratch<cvode_real_type_1d_view> scratch(member.team_scratch(level), per_team_extent);
        cvode_real_type_1d_view work(scratch.data(), per_team_extent);
        
        // Compute RHS using TChem's NetProductionRates
        if (kmcd.nConstSpec > 0) {
          TChem::Impl::NetProductionRates<Real, cvode_device_type>::team_invoke(
              member, temperature(i), pressure(i), vals_at_i, photo_rates_at_i,
              external_sources_at_i, const_tracers_at_i, rhs_at_i, work, kmcd);
        } else {
          TChem::Impl::NetProductionRates<Real, cvode_device_type>::team_invoke(
              member, temperature(i), pressure(i), vals_at_i, photo_rates_at_i,
              external_sources_at_i, rhs_at_i, work, kmcd);
        }
      });
  Kokkos::Profiling::popRegion();
  
  return 0;
}

// Jacobian function: J = df/dy
// This is called by CVODE to evaluate the Jacobian for all batches
int TChemATM::cvode_jac_func(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                              void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  auto* udata = static_cast<CVODEUserData*>(user_data);
  
  const auto nbatches = udata->nbatches;
  const auto batchSize = udata->batchSize;
  const auto policy = udata->policy;
  const auto kmcd = udata->kmcd;
  const auto temperature = udata->temperature;
  const auto pressure = udata->pressure;
  const auto const_tracers = udata->const_tracers;
  const auto photo_rates = udata->photo_rates;
  const auto external_sources = udata->external_sources;
  const auto fac = udata->fac;

  cvode_real_type_2d_view vals(N_VGetDeviceArrayPointer(y), nbatches, batchSize);
  auto J_data = sundials::kokkos::GetDenseMat<CVODEMatType>(J)->View();

  using problem_type = TChem::Impl::AtmosphericChemistryE3SM_Problem<Real, cvode_device_type>;

  const TChem::ordinal_type level = 1;
  const TChem::ordinal_type per_team_extent = problem_type::getWorkSpaceSize(kmcd) + batchSize;
  const std::string profile_name = "TChem::AtmosphericChemistryE3SM::CVODE_Jac";

#if defined(TCHEM_ATM_ENABLE_GPU)
  const auto JacRL = udata->JacRL;
#endif

  Kokkos::Profiling::pushRegion(profile_name);
  Kokkos::parallel_for(
      profile_name, policy,
      KOKKOS_LAMBDA(const typename cvode_policy_type::member_type& member) {
        const TChem::ordinal_type i = member.league_rank();
        
        const auto vals_at_i = Kokkos::subview(vals, i, Kokkos::ALL());
        const auto fac_at_i = Kokkos::subview(fac, i, Kokkos::ALL());
        const auto const_tracers_at_i = Kokkos::subview(const_tracers, i, Kokkos::ALL());
        
#if defined(TCHEM_ATM_ENABLE_GPU)
        const auto jacobian_at_i = Kokkos::subview(JacRL, i, Kokkos::ALL(), Kokkos::ALL());
#else
        const auto jacobian_at_i = Kokkos::subview(J_data, i, Kokkos::ALL(), Kokkos::ALL());
#endif
        
        cvode_real_type_1d_view photo_rates_at_i;
        if (photo_rates.extent(0) > 0) {
          photo_rates_at_i = Kokkos::subview(photo_rates, i, Kokkos::ALL());
        }
        
        cvode_real_type_1d_view external_sources_at_i;
        if (external_sources.extent(0) > 0) {
          external_sources_at_i = Kokkos::subview(external_sources, i, Kokkos::ALL());
        }
        
        // Get scratch memory and create views from raw pointer
        TChem::Scratch<cvode_real_type_1d_view> scratch(member.team_scratch(level), per_team_extent);
        auto wptr = scratch.data();
        
        const TChem::ordinal_type problem_workspace_size = problem_type::getWorkSpaceSize(kmcd);
        cvode_real_type_1d_view pw(wptr, problem_workspace_size);
        wptr += problem_workspace_size;
        
        problem_type problem;
        problem._kmcd = kmcd;
        problem._fac = fac_at_i;
        problem._work = pw;
        problem._temperature = temperature(i);
        problem._pressure = pressure(i);
        problem._const_concentration = const_tracers_at_i;
        problem._photo_rates = photo_rates_at_i;
        problem._external_sources = external_sources_at_i;
        
        problem.computeNumericalJacobian(member, vals_at_i, jacobian_at_i);
        
#if defined(TCHEM_ATM_ENABLE_GPU)
        // Copy from right-layout to left-layout for Sundials compatibility
        for (int k = 0; k < batchSize; ++k) {
          for (int j = 0; j < batchSize; ++j) {
            J_data(i, k, j) = jacobian_at_i(k, j);
          }
        }
#endif
      });
  Kokkos::Profiling::popRegion();
  
  return 0;
}

#endif  // TCHEM_ATM_ENABLE_SUNDIALS

TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;
  using namespace ShortFieldTagsNames;

  m_grid = m_grids_manager->get_grid("physics");
  EKAT_REQUIRE_MSG(m_grid != nullptr,
                   "Error! TChemATM could not get 'physics' grid.\n");

  const auto chem_file = m_params.get<std::string>(
      "chem_file", m_params.get<std::string>("chemfile", ""));
  EKAT_REQUIRE_MSG(!chem_file.empty(),
                   "Error! Missing required parameter 'chem_file' for tchem_atm.\n");

  const auto& grid_name = m_grid->name();
  const auto scalar3d_mid = m_grid->get_3d_scalar_layout(LEV);
  const FieldLayout scalar3d_int = m_grid->get_3d_scalar_layout(ILEV);
  const auto scalar2d = m_grid->get_2d_scalar_layout();
  add_field<Required>("p_mid", scalar3d_mid, Pa, grid_name);
  add_field<Required>("T_mid", scalar3d_mid, K, grid_name);
  add_field<Required>("qv", scalar3d_mid, q_unit, grid_name);
  add_field<Required>("p_int", scalar3d_int, Pa, grid_name);
  add_field<Required>("pseudo_density_dry", scalar3d_mid, Pa, grid_name);
  // Photo-table inputs (surface albedo, cloud and liquid)
  add_field<Required>("sfc_alb_dir_vis", scalar2d, none, grid_name);
  add_field<Required>("qc", scalar3d_mid, q_unit, grid_name);
  add_field<Required>("cldfrac_tot", scalar3d_mid, none, grid_name);


  // Build TChem kinetic model metadata from the configured chemistry file.
  if (m_atm_logger) m_atm_logger->debug("[TChemATM] KineticModelData");
  m_kmd = TChem::KineticModelData(chem_file);
  if (m_atm_logger) m_atm_logger->debug("[TChemATM] createNCAR_KineticModelConstData");
  m_kmcd = TChem::createNCAR_KineticModelConstData<tchem_device_type>(m_kmd);
  if (m_atm_logger) m_atm_logger->debug("[TChemATM] Done KineticModelData " + std::to_string(m_kmd.nSpec_));

  // Build m_species_mw indexed by TChem species order.
  // molecular_weights in the parameter list is a sublist mapping
  // species name -> MW (g/mol).
  m_species_mw.resize(m_kmd.nSpec_, 1.0);
  EKAT_REQUIRE_MSG(m_params.isSublist("molecular_weights"),
                   "Error! Missing required sublist 'molecular_weights' "
                   "under tchem_atm parameters.\n");
  const auto& mw_list = m_params.sublist("molecular_weights");
  const auto species_names_host = m_kmd.sNames_.view_host();
  //FIXME: get number of invariansts from chem mech. 
  m_num_invariants=9; // M, N2, O2, H2O, H2, and 4 constant tracers in the mechanism.
  for (int i = 0; i < m_kmcd.nSpec - m_num_invariants; ++i) {
    const std::string sname(&species_names_host(i, 0));
    EKAT_REQUIRE_MSG(mw_list.isParameter(sname),
                     "Error! Molecular weight not found for species '" +
                     sname + "' in 'molecular_weights' sublist.\n");
    m_species_mw[i] = mw_list.get<double>(sname);
  }
  if (m_atm_logger) m_atm_logger->debug("[TChemATM] Done loading molecular weights");
  m_tchem_ready = true;

  // Read sampling configuration for which atmospheric levels to run chemistry on.
  // Valid values: "troposphere" (default), "stratosphere", "all".
  m_chemistry_domain = m_params.get<std::string>("chemistry_domain", "troposphere");
  EKAT_REQUIRE_MSG(m_chemistry_domain == "troposphere" ||
                   m_chemistry_domain == "stratosphere" ||
                   m_chemistry_domain == "all",
                   "Error! Invalid 'chemistry_domain' value '" + m_chemistry_domain +
                   "'. Valid options: 'troposphere', 'stratosphere', 'all'.\n");
  if (m_atm_logger) m_atm_logger->info("[TChemATM] chemistry_domain = " + m_chemistry_domain);

  //FIXME: invariants are not tracers.
  for (int i = 0; i < m_kmd.nSpec_ - m_num_invariants; ++i) {
    const std::string sname(&species_names_host(i, 0));
    add_tracer<Updated>(sname, m_grid, q_unit);
  }
  if (m_atm_logger) m_atm_logger->info("[TChemATM] Number of tracers added: " + std::to_string(m_kmd.nSpec_ - m_num_invariants));
    // Add prescribed constant tracer fields (oxidants).
  // M, N2, O2, H2O, H2, CH4 are computed from T and P at runtime, not registered as fields.
  for (int j = 0; j < s_num_cnst_tracers; ++j) {
    const std::string sname(&species_names_host(m_kmcd.M_index + 6 + j, 0));
    add_field<Updated>(sname, scalar3d_mid, q_unit, grid_name);
  }
}

void TChemATM::initialize_impl(const RunType /* run_type */) {
  EKAT_REQUIRE_MSG(m_tchem_ready,
                   "Error! TChemATM::initialize_impl called before TChem model initialization.\n");

  m_ncols = m_grid->get_num_local_dofs();
  m_nlevs = m_grid->get_num_vertical_levels();
  m_nbatch = m_ncols * m_nlevs;
  if (m_nbatch == 0) {
    return;
  }

  // Match MAM behavior: cache direct visible surface albedo view once.
  m_sfc_alb_dir_vis = get_field_in("sfc_alb_dir_vis").get_view<const Real *>();

  // Cache column lat/lon from grid geometry (constant for the whole run).
  // Convert from degrees to radians here so run_impl does no conversion per step.
  {
    const auto lat_deg = m_grid->get_geometry_data("lat").get_view<const Real *, Host>();
    const auto lon_deg = m_grid->get_geometry_data("lon").get_view<const Real *, Host>();
    m_col_latitudes_rad  = host_view_1d("tchem_lat_rad",  m_ncols);
    m_col_longitudes_rad = host_view_1d("tchem_lon_rad", m_ncols);
    for (int i = 0; i < m_ncols; ++i) {
      m_col_latitudes_rad(i)  = lat_deg(i) * M_PI / 180.0;
      m_col_longitudes_rad(i) = lon_deg(i) * M_PI / 180.0;
    }
  }

  m_n_active_vars      = m_kmcd.nSpec - m_kmcd.nConstSpec;
  m_state_vec_dim      = TChem::Impl::getStateVectorSize(m_kmcd.nSpec);
  m_species_names_host = m_kmd.sNames_.view_host();


  m_state = explicit_euler_type::real_type_2d_view_type("tchem_state", m_nbatch, m_state_vec_dim);
  const int m_photo_reactions = mam4::mo_photo::phtcnt;
  m_photo_rates = explicit_euler_type::real_type_2d_view_type("tchem_photo_rates", m_nbatch, m_photo_reactions);
  m_external_sources = explicit_euler_type::real_type_2d_view_type("tchem_external_sources", m_nbatch, m_n_active_vars);
  m_t = explicit_euler_type::real_type_1d_view_type("tchem_time", m_nbatch);
  m_dt_view = explicit_euler_type::real_type_1d_view_type("tchem_dt", m_nbatch);
  m_tadv = TChem::time_advance_type_1d_view("tchem_tadv", m_nbatch);

  // Temporary views for tropopause computation
  m_dz         = view_2d("tchem_dz",         m_ncols, m_nlevs);
  m_z_iface    = view_2d("tchem_z_iface",    m_ncols, m_nlevs + 1);
  m_z_mid      = view_2d("tchem_z_mid",      m_ncols, m_nlevs);
  m_qv_dry     = view_2d("tchem_qv_dry",     m_ncols, m_nlevs);
  m_zenith_angle = view_1d("tchem_zenith_angle", m_ncols);
  // Pre-allocate host mirror for zenith angle to avoid per-timestep allocation.
  // Note: shr_orb_cosz_c2f is a Fortran routine that must run on the host.
  m_zenith_angle_host = host_view_1d("tchem_zenith_angle_host", m_ncols);
  m_ilev_tropp = view_1d_int("tchem_ilev_tropp", m_ncols);
  // Allocate persistent index/offset views once here and reuse in run_impl.
  m_offsets = view_1d_int("tchem_offsets", m_ncols + 1);
  m_sample_icol = view_1d_int("tchem_sample_icol", m_nbatch);
  m_sample_ilev = view_1d_int("tchem_sample_ilev", m_nbatch);
  // Read solver/time-stepping parameters from the namelist.
  m_solver_type          = m_params.get<std::string>("solver_type", "implicit_euler");
  if (m_solver_type == "implicit_euler") {
    m_solver_enum = SolverType::ImplicitEuler;
  } else if (m_solver_type == "trbdf2") {
    m_solver_enum = SolverType::TRBDF2;
  } else if (m_solver_type == "explicit_euler") {
    m_solver_enum = SolverType::ExplicitEuler;
  } else if (m_solver_type == "cvode_batch") {
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
    m_solver_enum = SolverType::CVODEBatch;
#else
    EKAT_REQUIRE_MSG(false, "Error! solver_type 'cvode_batch' requires TCHEM_ATM_ENABLE_SUNDIALS=ON.\n");
#endif
  } else {
    EKAT_REQUIRE_MSG(false, "Error! Unknown solver_type '" + m_solver_type +
                     "'. Valid options: implicit_euler, trbdf2, explicit_euler, cvode_batch.\n");
  }
  m_max_time_iterations    = m_params.get<int>("max_time_iterations", 100);
  m_max_newton_iterations  = m_params.get<int>("max_newton_iterations", 100);
  m_jacobian_interval      = m_params.get<int>("jacobian_interval", 1);
  m_dtmin_sub              = m_params.get<double>("dtmin_sub", 1e-1);
  m_dtmax_sub              = m_params.get<double>("dtmax_sub", -1.0);
  m_atol_newton            = m_params.get<double>("atol_newton", 1e-10);
  m_rtol_newton            = m_params.get<double>("rtol_newton", 1e-6);
  m_atol_time              = m_params.get<double>("atol_time", 1e-12);
  m_rtol_time              = m_params.get<double>("rtol_time", 1e-4);
  m_use_shared_workspace   = m_params.get<bool>("use_shared_workspace", true);
  m_orbital_year           = m_params.get<int>("orbital_year", -9999);
  m_orbital_eccen          = m_params.get<double>("orbital_eccentricity", -9999.0);
  m_orbital_obliq          = m_params.get<double>("orbital_obliquity", -9999.0);
  m_orbital_mvelp          = m_params.get<double>("orbital_mvelp", -9999.0);
  if (m_atm_logger) m_atm_logger->info("[TChemATM] solver_type = " + m_solver_type);

  // Allocate and populate tolerance/scaling views for implicit solvers.
  if (m_solver_enum == SolverType::ImplicitEuler || m_solver_enum == SolverType::TRBDF2) {
    using problem_type =
        TChem::Impl::AtmosphericChemistryE3SM_Problem<TChem::real_type,
                                                      tchem_device_type>;
    const TChem::ordinal_type number_of_equations =
        problem_type::getNumberOfTimeODEs(m_kmcd);

    m_tol_newton = explicit_euler_type::real_type_1d_view_type("tchem_tol_newton", 2);
    m_tol_time   = explicit_euler_type::real_type_2d_view_type("tchem_tol_time",
                                                                number_of_equations, 2);
    m_fac        = explicit_euler_type::real_type_2d_view_type("tchem_fac",
                                                                m_nbatch, number_of_equations);

    auto tol_newton_host = Kokkos::create_mirror_view(m_tol_newton);
    auto tol_time_host   = Kokkos::create_mirror_view(m_tol_time);
    tol_newton_host(0) = m_atol_newton;
    tol_newton_host(1) = m_rtol_newton;
    for (TChem::ordinal_type i = 0; i < number_of_equations; ++i) {
      tol_time_host(i, 0) = m_atol_time;
      tol_time_host(i, 1) = m_rtol_time;
    }
    Kokkos::deep_copy(m_tol_newton, tol_newton_host);
    Kokkos::deep_copy(m_tol_time, tol_time_host);
    Kokkos::deep_copy(m_fac, 0.0);
  }

  if (!m_use_shared_workspace) {
    TChem::ordinal_type per_team_extent = 0;
    if (m_solver_enum == SolverType::ImplicitEuler || m_solver_enum == SolverType::TRBDF2) {
      per_team_extent = TChem::AtmosphericChemistryE3SM::getWorkSpaceSize(m_kmcd);
    } else {
      per_team_extent = TChem::AtmosphericChemistryE3SM_ExplicitEuler::getWorkSpaceSize(m_kmcd);
    }
    m_workspace = explicit_euler_type::real_type_2d_view_type(
        "tchem_workspace", m_nbatch, per_team_extent);
  }

  // Read CVODE-specific parameters from namelist (under cvode_parameters sublist)
  if (m_params.isSublist("cvode_parameters")) {
    const auto& cvode_params = m_params.sublist("cvode_parameters");
    m_cvode_rtol = cvode_params.get<double>("rtol", 1e-8);
    m_cvode_atol = cvode_params.get<double>("atol", 1e-12);
    m_cvode_max_steps = cvode_params.get<int>("max_steps", 10000);
    m_cvode_max_step = cvode_params.get<double>("max_step", 0.0);
    m_cvode_min_step = cvode_params.get<double>("min_step", 0.0);
  }

  // CVODE batch solver initialization
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_solver_enum == SolverType::CVODEBatch) {
    using problem_type = TChem::Impl::AtmosphericChemistryE3SM_Problem<Real, cvode_device_type>;
    const TChem::ordinal_type number_of_equations = problem_type::getNumberOfTimeODEs(m_kmcd);
    
    if (m_atm_logger) m_atm_logger->info("[TChemATM] Initializing CVODE batch solver");
    
    // Create Sundials context
    m_sundials_ctx = std::make_unique<sundials::Context>();
    
    // Allocate CVODE vectors
    CVODESizeType length{static_cast<CVODESizeType>(m_nbatch * number_of_equations)};
    m_cvode_y = std::make_unique<CVODEVecType>(length, *m_sundials_ctx);
    m_cvode_abstol = std::make_unique<CVODEVecType>(length, *m_sundials_ctx);
    N_VConst(SUN_RCONST(m_cvode_atol), *m_cvode_abstol);
    
    // Create CVODE memory using BDF methods (suitable for stiff problems)
    m_cvode_mem = CVodeCreate(CV_BDF, *m_sundials_ctx);
    EKAT_REQUIRE_MSG(m_cvode_mem != nullptr, "Error! CVodeCreate failed.\n");
    
    // Initialize with t0=0, y0 will be set in run_impl
    int retval = CVodeInit(m_cvode_mem, cvode_rhs_func, SUN_RCONST(0.0), *m_cvode_y);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeInit failed with code " + std::to_string(retval) + ".\n");
    
    // Set tolerances
    retval = CVodeSVtolerances(m_cvode_mem, SUN_RCONST(m_cvode_rtol), *m_cvode_abstol);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSVtolerances failed.\n");
    
    // Set maximum number of internal steps
    retval = CVodeSetMaxNumSteps(m_cvode_mem, m_cvode_max_steps);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetMaxNumSteps failed.\n");
    
    // Set step size limits if specified (0 means use CVODE defaults)
    if (m_cvode_max_step > 0.0) {
      retval = CVodeSetMaxStep(m_cvode_mem, SUN_RCONST(m_cvode_max_step));
      EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetMaxStep failed.\n");
    }
    if (m_cvode_min_step > 0.0) {
      retval = CVodeSetMinStep(m_cvode_mem, SUN_RCONST(m_cvode_min_step));
      EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetMinStep failed.\n");
    }
    
    // Allocate user data views
    m_cvode_temperature = cvode_real_type_1d_view("cvode_temperature", m_nbatch);
    m_cvode_pressure = cvode_real_type_1d_view("cvode_pressure", m_nbatch);
    m_cvode_const_tracers = cvode_real_type_2d_view("cvode_const_tracers", m_nbatch, m_kmcd.nConstSpec);
    
    // Set up user data structure
    m_cvode_udata.nbatches = m_nbatch;
    m_cvode_udata.batchSize = number_of_equations;
    m_cvode_udata.kmcd = m_kmcd;
    m_cvode_udata.temperature = m_cvode_temperature;
    m_cvode_udata.pressure = m_cvode_pressure;
    m_cvode_udata.const_tracers = m_cvode_const_tracers;
    m_cvode_udata.photo_rates = m_photo_rates;
    m_cvode_udata.external_sources = m_external_sources;
    m_cvode_udata.fac = cvode_real_type_2d_view("cvode_fac", m_nbatch, number_of_equations);
    Kokkos::deep_copy(m_cvode_udata.fac, 0.0);
    
#if defined(TCHEM_ATM_ENABLE_GPU)
    m_cvode_udata.JacRL = cvode_real_type_3d_view("cvode_JacRL", m_nbatch, number_of_equations, number_of_equations);
#endif
    
    // Attach user data
    retval = CVodeSetUserData(m_cvode_mem, &m_cvode_udata);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetUserData failed.\n");
    
    // Set up dense direct linear solver with user-supplied Jacobian
    m_cvode_A = std::make_unique<CVODEMatType>(m_nbatch, number_of_equations, number_of_equations, *m_sundials_ctx);
    m_cvode_LS = std::make_unique<CVODELSType>(*m_sundials_ctx);
    
    retval = CVodeSetLinearSolver(m_cvode_mem, m_cvode_LS->Convert(), m_cvode_A->Convert());
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetLinearSolver (dense) failed.\n");
    
    retval = CVodeSetJacFn(m_cvode_mem, cvode_jac_func);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeSetJacFn failed.\n");
    
    const TChem::ordinal_type per_team_extent_cvode = problem_type::getWorkSpaceSize(m_kmcd) + number_of_equations;
    
    // Set up team policy for CVODE callbacks
    m_cvode_udata.policy = cvode_policy_type(TChem::exec_space(), m_nbatch, Kokkos::AUTO());
    const TChem::ordinal_type per_team_scratch =
        TChem::Scratch<cvode_real_type_1d_view>::shmem_size(per_team_extent_cvode);
    m_cvode_udata.policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));
    
    if (m_atm_logger) m_atm_logger->info("[TChemATM] CVODE batch solver initialized successfully");
  }
#endif  // TCHEM_ATM_ENABLE_SUNDIALS

  // Photo table initialization (optional)
  const std::string rsf_file = m_params.get<std::string>("mam4_rsf_file", "");
  const std::string xs_long_file = m_params.get<std::string>("mam4_xs_long_file", "");
  if (!rsf_file.empty() && !xs_long_file.empty()) {
    m_photo_table = tchem::read_photo_table_uci(rsf_file, xs_long_file);
    m_photo_table_len = mam4::mo_photo::get_photo_table_work_len(m_photo_table);
    m_work_photo_table = view_2d("tchem_photo_work", m_ncols, m_photo_table_len);
    // allocate a 3D photo buffer: (ncols, nlevs, phtcnt)
    m_photo_3d = view_3d("tchem_photo_3d", m_ncols, m_nlevs, mam4::mo_photo::phtcnt);
    // allocate O3 column buffer
    m_o3col_dens = view_2d("tchem_o3col_dens", m_ncols, m_nlevs);
    // find O3 species index in kinetic model names (if present)
    m_o3_species_index = -1;
    for (int i = 0; i < m_kmd.nSpec_; ++i) {
      const std::string sname(&m_species_names_host(i, 0));
      if (sname == "O3") { m_o3_species_index = i; break; }
    }
    m_have_photo_table = true;
  } else {
    m_have_photo_table = false;
  }

  set_exo_coldens_reader();
}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::set_exo_coldens_reader() {
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  // Check if exo coldens is requested (optional feature)
  const std::string exo_coldens_file_name =
      m_params.get<std::string>("mam4_exo_coldens_file_name", "");
  if (exo_coldens_file_name.empty()) {
    m_have_exo_coldens = false;
    return;
  }

  const auto pint = get_field_in("p_int");
  // Exo column density fields read initialization
  const std::string exo_coldens_map_file =
      m_params.get<std::string>("aero_microphys_remap_file", "");
  const auto exo_coldens_time_interpolation_method = 
      m_params.get<std::string>("time_interpolation_method", "yearly_periodic");

  // get fields from FM.
  auto grid_exo_coldens = m_grid->clone("exo_grid", true);
  grid_exo_coldens->reset_vertical_configuration(1, AbstractGrid::VKind::Model);
  auto layout = grid_exo_coldens->get_3d_scalar_layout(LEV);

  auto molec = none;
  auto cm2 = pow(m / 100, 2);
  auto molec_cm2 = Units(molec / cm2, "molecules/cm2");
  const std::string exo_coldens_name = "O3_column_density";
  Field field_exo(
      FieldIdentifier(exo_coldens_name, layout, molec_cm2, grid_exo_coldens->name()));
  field_exo.allocate_view();

  m_exo_coldens_fields.clear();
  m_exo_coldens_fields.push_back(field_exo);

  m_data_interp_exo_coldens = 
      std::make_shared<DataInterpolation>(grid_exo_coldens, m_exo_coldens_fields);
  m_data_interp_exo_coldens->setup_time_database(
      {exo_coldens_file_name}, exo_coldens_time_interpolation_method);
  m_data_interp_exo_coldens->create_horiz_remappers(
      exo_coldens_map_file == "none" ? "" : exo_coldens_map_file, m_iop_data_manager);
  m_data_interp_exo_coldens->set_logger(m_atm_logger);

  DataInterpolation::VertRemapData remap_exo_coldens;
  remap_exo_coldens.vr_type = DataInterpolation::Custom;
  // We are using a custom remapper that invokes the MAM4XX routine
  // for vertical interpolation.
  auto grid_after_hremap = m_data_interp_exo_coldens->get_grid_after_hremap();
  auto vertical_remapper = 
      std::make_shared<VerticalRemapperExoColdensMAM4>(grid_after_hremap, grid_exo_coldens);
  vertical_remapper->set_delta_pressure(exo_coldens_file_name, pint);
  remap_exo_coldens.custom_remapper = vertical_remapper;

  m_data_interp_exo_coldens->create_vert_remapper(remap_exo_coldens);
  m_data_interp_exo_coldens->init_time_interpolation(
      start_of_step_ts(), DataInterpolation::Linear);
  m_have_exo_coldens = true;
}

void TChemATM::run_impl(const double dt) {

  EKAT_ASSERT_MSG(m_tchem_ready,
                   "Error! TChemATM::run_impl called before TChem model initialization.\n");
 
  using ordinal_type = TChem::ordinal_type;

  if (m_nbatch == 0) {
    return;
  }

  if (m_have_exo_coldens && m_data_interp_exo_coldens) {
    m_data_interp_exo_coldens->run(end_of_step_ts());
  }

  const auto& t_mid = get_field_in("T_mid").get_view<const Real **>();
  const auto& p_mid = get_field_in("p_mid").get_view<const Real **>();
  const auto& qv = get_field_in("qv").get_view<const Real **>();
  const int nlevs = m_nlevs;
  const auto state = m_state;

  const auto& p_int     = get_field_in("p_int").get_view<const Real **>();
  const auto& p_del_dry = get_field_in("pseudo_density_dry").get_view<const Real **>();
  const int ncols = m_ncols;

  const auto& dz         = m_dz;
  const auto& z_iface    = m_z_iface;
  const auto& z_mid      = m_z_mid;
  const auto& qv_dry     = m_qv_dry;
  const auto& ilev_tropp = m_ilev_tropp;

  using TPF = ekat::TeamPolicyFactory<KT::ExeSpace>;
  using PF  = scream::PhysicsFunctions<DefaultDevice>;
  const auto col_policy = TPF::get_default_team_policy(ncols, nlevs);
  // Policy for kernels containing parallel_scan (requires power-of-2 team size on CUDA)
  const auto scan_policy = TPF::get_thread_range_parallel_scan_team_policy(ncols, nlevs);
  const Real z_surf = 0.0;

  // Compute dry water vapor mass mixing ratio
  Kokkos::parallel_for(
    "tchem_qv_dry", col_policy,
    KOKKOS_LAMBDA(const ThreadTeam& team) {
      const int icol = team.league_rank();
      Kokkos::parallel_for(Kokkos::TeamVectorRange(team, nlevs), [&](int kk) {
        qv_dry(icol, kk) =
            PF::calculate_drymmr_from_wetmmr(qv(icol, kk), qv(icol, kk));
      });
    });

  // Compute layer thickness from dry pseudo-density
  Kokkos::parallel_for(
    "tchem_dz", col_policy,
    KOKKOS_LAMBDA(const ThreadTeam& team) {
      const int icol = team.league_rank();
      PF::calculate_dz(team, ekat::subview(p_del_dry, icol),
                       ekat::subview(p_mid, icol), ekat::subview(t_mid, icol),
                       ekat::subview(qv_dry, icol), ekat::subview(dz, icol));
    });

  // Compute interface geopotential heights.
  // Uses parallel_scan internally, so requires scan_policy on CUDA.
  Kokkos::parallel_for(
    "tchem_z_int", scan_policy,
    KOKKOS_LAMBDA(const ThreadTeam& team) {
      const int icol = team.league_rank();
      PF::calculate_z_int(team, nlevs, ekat::subview(dz, icol),
                          z_surf, ekat::subview(z_iface, icol));
    });

  // Compute midpoint geopotential heights.
  // Uses parallel_scan internally, so requires scan_policy on CUDA.
  Kokkos::parallel_for(
    "tchem_z_mid", scan_policy,
    KOKKOS_LAMBDA(const ThreadTeam& team) {
      const int icol = team.league_rank();
      PF::calculate_z_mid(team, nlevs, ekat::subview(z_iface, icol),
                          ekat::subview(z_mid, icol));
    });

  // Compute tropopause level per column using the Reichler et al. [2003] algorithm
  Kokkos::parallel_for(
    "tchem_tropopause",
    Kokkos::RangePolicy<KT::ExeSpace>(0, ncols),
    KOKKOS_LAMBDA(const int icol) {
      ilev_tropp(icol) = mam4::aero_rad_props::tropopause_or_quit(
          ekat::subview(p_mid,   icol), ekat::subview(p_int,   icol),
          ekat::subview(t_mid,   icol), ekat::subview(z_mid,   icol),
          ekat::subview(z_iface, icol));
    });


  const auto& ntropopause = ilev_tropp; // number of levels up to tropopause
  const int ncol = ncols;
  const int nlev = nlevs;

  // Determine which levels to run chemistry on based on chemistry_domain.
  // "troposphere": levels >= tropopause index (below tropopause)
  // "stratosphere": levels < tropopause index (above tropopause)
  // "all": all levels
  const bool run_all_levels = (m_chemistry_domain == "all");
  const bool above = (m_chemistry_domain == "troposphere");
  
  if (run_all_levels) {
    m_nsamples = ncol * nlev;
  } else {
    m_nsamples = tchem::compute_nsamples(ntropopause, ncol, nlev, above);
  }
  if (m_atm_logger) m_atm_logger->info("[TChemATM] m_nsamples = " + std::to_string(m_nsamples));

  using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
 
  policy_type policy(TChem::exec_space(), m_nsamples, Kokkos::AUTO());
  ordinal_type per_team_extent = 0;
  if (m_solver_enum == SolverType::ImplicitEuler || m_solver_enum == SolverType::TRBDF2) {
    per_team_extent = TChem::AtmosphericChemistryE3SM::getWorkSpaceSize(m_kmcd);
  } else {
    per_team_extent = TChem::AtmosphericChemistryE3SM_ExplicitEuler::getWorkSpaceSize(m_kmcd);
  }
  //TODO: add the workspace for implicit_euler
  // use the use_shared_workspace option to turn on and offf
  if (m_use_shared_workspace) {
    const ordinal_type per_team_scratch =
        TChem::Scratch<TChem::real_type_1d_view>::shmem_size(per_team_extent);
    policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));
  }

  // Compute offsets and sampled indices used by state/photo packing.
  if (run_all_levels) {
    // For "all" domain, indices are simply column-major: sample = icol * nlev + ilev
    const auto sample_icol = m_sample_icol;
    const auto sample_ilev = m_sample_ilev;
    Kokkos::parallel_for(
        "tchem_fill_all_indices", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
        KOKKOS_LAMBDA(const int isample) {
          sample_icol(isample) = isample / nlev;
          sample_ilev(isample) = isample % nlev;
        });
  } else {
    tchem::compute_offsets(ntropopause, ncol, nlev, m_offsets, above);
    tchem::compute_sample_indices(ntropopause, m_offsets, ncol, nlev, m_sample_icol,
                                  m_sample_ilev, above);
  }

  // Compute photo table rates if we have a photo table
 if (m_have_photo_table) {
    // Zero photo_rates before computing - must zero full m_nbatch to ensure
   // no stale data when m_nsamples < m_nbatch (troposphere/stratosphere selection).
    Kokkos::deep_copy(m_photo_rates, 0.0);

    // Compute orbital eccentricity factor used by MAM photo_table.
    int orbital_year = m_orbital_year;
    double eccen = m_orbital_eccen;
    double obliq = m_orbital_obliq;
    double mvelp = m_orbital_mvelp;
    double obliqr, lambm0, mvelpp;
    if (eccen >= 0 && obliq >= 0 && mvelp >= 0) {
      orbital_year = shr_orb_undef_int_c2f;
    } else if (orbital_year < 0) {
      orbital_year = start_of_step_ts().get_year();
    }
    shr_orb_params_c2f(&orbital_year, &eccen, &obliq, &mvelp,
                       &obliqr, &lambm0, &mvelpp);
    const auto calday = start_of_step_ts().frac_of_year_in_days() + 1;
    double delta = 0, eccf = 1.0;
    shr_orb_decl_c2f(calday, eccen, mvelpp, lambm0, obliqr, &delta, &eccf);

    // Compute zenith angle on host using pre-allocated view, then copy to device.
    // Note: shr_orb_cosz_c2f is a Fortran routine that cannot run on GPU.
    // The host view m_zenith_angle_host is pre-allocated in initialize_impl
    // to avoid per-timestep allocation overhead.
    for (int i = 0; i < m_ncols; ++i) {
      const Real cosz = shr_orb_cosz_c2f(calday, m_col_latitudes_rad(i),
                                          m_col_longitudes_rad(i), delta, dt);
      m_zenith_angle_host(i) = std::acos(cosz);
    }
    Kokkos::deep_copy(m_zenith_angle, m_zenith_angle_host);

    // Zero the 3D photo buffer before table_photo fills it
    Kokkos::deep_copy(m_photo_3d, 0.0);

    // Prepare inputs for photo table computation
    const auto& sfc_alb = m_sfc_alb_dir_vis;
    const auto& zenith_angle = m_zenith_angle;
    const auto& qc_field = get_field_in("qc").get_view<const Real **>();
    const auto& cldfrac = get_field_in("cldfrac_tot").get_view<const Real **>();

    view_2d o3_field;
    bool have_o3_field = false;
    if (m_o3_species_index >= 0) {
      const std::string o3_name(&m_species_names_host(m_o3_species_index, 0));
      o3_field = get_field_out(o3_name).get_view<Real **>();
      have_o3_field = true;
    }

    // ozone column buffer (preallocated in initialize_impl)
    // Zero o3 column density - only written when have_o3_field is true
    Kokkos::deep_copy(m_o3col_dens, 0.0);
    view_2d o3_exo_col;
    const bool have_o3_exo_col = m_have_exo_coldens && !m_exo_coldens_fields.empty();
    if (have_o3_exo_col) {
      o3_exo_col = m_exo_coldens_fields[0].get_view<Real **>();
    }
    const auto work_photo_table = m_work_photo_table;
    const auto photo_table = m_photo_table;
    const auto o3_col_dens = m_o3col_dens ;
    const auto photo_3d = m_photo_3d;
    const Real o3_species_mw =
        have_o3_field ? m_species_mw[m_o3_species_index] : 0.0;
    Kokkos::parallel_for(
        "tchem_photo_table", col_policy,
        KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int icol = team.league_rank();
          // per-column work array (1D view)
          const auto work_photo_table_icol = ekat::subview(work_photo_table, icol);
          mam4::mo_photo::PhotoTableWorkArrays photo_work_arrays_icol;
          mam4::mo_photo::set_photo_table_work_arrays(photo_table, work_photo_table_icol,
                                                     photo_work_arrays_icol);
          team.team_barrier();
          // subviews for column inputs
          const auto pmid_col = ekat::subview(p_mid, icol);
          const auto pdel_col = ekat::subview(p_del_dry, icol);
          const auto t_col = ekat::subview(t_mid, icol);
          //CHECK: compare against microphysic interface and check how
          // o3_col is computed there and here.  
          const auto o3_icol = ekat::subview(o3_col_dens, icol);
          // compute o3 column densities if we have an O3 tracer
          if (have_o3_field) {
            const auto mmr_o3_col = ekat::subview(o3_field, icol);
            const Real o3_exo_top = have_o3_exo_col ? o3_exo_col(icol, 0) : 0.0;
            // compute column densities (molecules/cm^2) from mmr and pdel
            mam4::microphysics::compute_o3_column_density(team, pdel_col, mmr_o3_col,
                                                          o3_exo_top, o3_species_mw,
                                                          o3_icol);
          }
          const Real srfalb = sfc_alb(icol);
          const auto qc_col = ekat::subview(qc_field, icol);
          const auto cld_col = ekat::subview(cldfrac, icol);
          const auto photo_icol = ekat::subview(photo_3d, icol);
          const Real esfact = eccf;
          mam4::mo_photo::table_photo(team, photo_icol, pmid_col, pdel_col, t_col,
                                     o3_icol, zenith_angle(icol), srfalb, qc_col, cld_col,
                                     esfact, photo_table, photo_work_arrays_icol);
        });

    // Copy sampled levels into m_photo_rates (nsamples x nphoto_reactions).
    const int nphoto_reactions =
        std::min(static_cast<int>(m_photo_rates.extent(1)),
                 mam4::mo_photo::phtcnt);
    tchem::pack_photo_rates_into_state(
        m_photo_rates, m_photo_3d, m_sample_icol, m_sample_ilev, m_nsamples,
        nphoto_reactions, "tchem_photo_copy");
    // Note: No fence needed here - Kokkos execution order ensures data is ready
    // before the init kernel uses m_photo_rates.
  }


  // Initialize solver state arrays.
  // NOTE: Must use deep_copy for full m_nbatch extent to ensure BFB results
  // when m_nsamples < m_nbatch (troposphere/stratosphere selection).
  Kokkos::deep_copy(m_external_sources, 0.0);
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
  tadv_default._max_num_newton_iterations = m_max_newton_iterations;
  tadv_default._num_time_iterations_per_interval = 100;
  tadv_default._jacobian_interval = m_jacobian_interval;
  Kokkos::deep_copy(m_tadv, tadv_default);

  // Create local copies for device lambdas (avoid implicit this capture)
  const auto t_view = m_t;
  const auto dt_view_local = m_dt_view;
  const auto tadv = m_tadv;

  // Pack pressure and temperature into the state for all selected samples
  tchem::pack_into_state(state, p_mid, m_sample_icol, m_sample_ilev, m_nsamples, 1,
                  "tchem_init_state_p");
  tchem::pack_into_state(state, t_mid, m_sample_icol, m_sample_ilev, m_nsamples, 2,
                  "tchem_init_state_t");

  for (int ivar = 0; ivar < m_n_active_vars; ++ivar) {
    const auto& tracer_name = std::string(&m_species_names_host(ivar, 0));
    // std::cout << "[TChemATM] Filling state column for tracer " << tracer_name << "\n";
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    // Use sampling-aware pack to only pack selected samples (m_sample_icol/ilev)
    tchem::pack_wet_mmr_into_state(state, q_tracer, qv, m_sample_icol, m_sample_ilev,
                m_nsamples, ivar + 3, m_species_mw[ivar],
                "tchem_init_state_tracer");
  }
  // conversion factor for Pascals to dyne/cm^2
  constexpr Real Pa_xfac = 10.0;
  // presumably, the boltzmann constant, in CGS units
  constexpr Real boltz_cgs = 0.13806500000000001E-015;
  // Compute invariants:
  // step 1: M [molecules/cm^3] = Pa_xfac * P [Pa] / (boltz_cgs * T [K])

  // invariant_col: first column index in state[] where invariants are stored
  const int invariant_col = m_kmcd.M_index + 3;
  const auto sample_icol = m_sample_icol;
  const auto sample_ilev = m_sample_ilev;
  Kokkos::parallel_for(
      "tchem_compute_M", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
      KOKKOS_LAMBDA(const int isample) {
        const int icol = sample_icol(isample);
        const int ilev = sample_ilev(isample);
        const Real M_value =
            Pa_xfac * p_mid(icol, ilev) / (boltz_cgs * t_mid(icol, ilev));
        state(isample, invariant_col) = M_value;
        // N2 = 0.79 * M
        state(isample, invariant_col + 1) = 0.79;  // * M_value;
        // O2 = 0.21 * M
        state(isample, invariant_col + 2) = 0.21;  // * M_value;
        // H2O = qv * M / (1 + qv)
        state(isample, invariant_col + 3) =
            qv(icol, ilev) / (1.0 + qv(icol, ilev));  // M_value;
        // H2 = 5.5e-7 * M
        state(isample, invariant_col + 4) = 5.5e-7;  // * M_value;
        // CH4 = 0
        state(isample, invariant_col + 5) = 0.0;
      });


  for (int j = 0; j < s_num_cnst_tracers; ++j) {
    const auto& tracer_name = std::string(&m_species_names_host(m_kmcd.M_index + 6 + j, 0));
    const auto& q_tracer = get_field_out(tracer_name).get_view<Real **>();
    const int state_col_j = invariant_col + 6 + j;
    tchem::pack_into_state(state, q_tracer, m_sample_icol, m_sample_ilev,
                           m_nsamples, state_col_j,
                           "tchem_compute_cnst_tracer");
  }
  // Time loop: mirrors TChem_AtmosphericChemistryE3SM.cpp standalone example.
  // Solver type and time-stepping parameters are controlled via namelist.
  // Note: tadv, t_view, dt_view_local are already defined above for init kernel.

  // ============================================================================
  // Time Integration - Solver-specific paths
  // ============================================================================
  
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_solver_enum == SolverType::CVODEBatch) {
    // ========================================================================
    // CVODE Batch Solver Path
    // ========================================================================
    using cvode_problem_type = TChem::Impl::AtmosphericChemistryE3SM_Problem<Real, cvode_device_type>;
    const TChem::ordinal_type number_of_equations = cvode_problem_type::getNumberOfTimeODEs(m_kmcd);
    
    // Update user data views for this timestep
    // Copy temperature and pressure from sampled state to CVODE views
    const auto cvode_temperature = m_cvode_temperature;
    const auto cvode_pressure = m_cvode_pressure;
    const auto cvode_const_tracers = m_cvode_const_tracers;
    const auto nConstSpec = m_kmcd.nConstSpec;
    
    Kokkos::parallel_for(
        "cvode_copy_T_P", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
        KOKKOS_LAMBDA(const int isample) {
          const int icol = sample_icol(isample);
          const int ilev = sample_ilev(isample);
          cvode_temperature(isample) = t_mid(icol, ilev);
          cvode_pressure(isample) = p_mid(icol, ilev);
        });
    
    // Copy constant tracers from state to CVODE const_tracers view
    const int const_tracer_start = m_n_active_vars + 3;  // After P, T, density and active vars
    Kokkos::parallel_for(
        "cvode_copy_const_tracers", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < nConstSpec; ++j) {
            cvode_const_tracers(isample, j) = state(isample, const_tracer_start + j);
          }
        });
    
    // Wrap CVODE y vector as 2D view and copy active species from state
    cvode_real_type_2d_view y2d(m_cvode_y->View().data(), m_nsamples, number_of_equations);
    
    Kokkos::parallel_for(
        "cvode_pack_y", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < number_of_equations; ++j) {
            y2d(isample, j) = state(isample, j + 3);  // Skip P, T, density in state
          }
        });
    
    // Update UserData with current photo_rates and external_sources
    m_cvode_udata.photo_rates = m_photo_rates;
    m_cvode_udata.external_sources = m_external_sources;
    m_cvode_udata.nbatches = m_nsamples;  // Use actual sample count, not full batch
    
    // Update policy for current sample count
    m_cvode_udata.policy = cvode_policy_type(TChem::exec_space(), m_nsamples, Kokkos::AUTO());
    const TChem::ordinal_type cvode_per_team_extent = 
        cvode_problem_type::getWorkSpaceSize(m_kmcd) + number_of_equations;
    const TChem::ordinal_type cvode_per_team_scratch =
        TChem::Scratch<cvode_real_type_1d_view>::shmem_size(cvode_per_team_extent);
    m_cvode_udata.policy.set_scratch_size(1, Kokkos::PerTeam(cvode_per_team_scratch));
    
    // Reinitialize CVODE for this timestep (t0=0, y0 = current y)
    int retval = CVodeReInit(m_cvode_mem, SUN_RCONST(0.0), *m_cvode_y);
    EKAT_REQUIRE_MSG(retval >= 0, "Error! CVodeReInit failed.\n");
    
    // Time integration using CVODE - advance from t=0 to t=dt
    const sunrealtype Tf = SUN_RCONST(dt);
    sunrealtype t_cvode = SUN_RCONST(0.0);
    
    // Advance CVODE to final time
    retval = CVode(m_cvode_mem, Tf, *m_cvode_y, &t_cvode, CV_NORMAL);
    
    if (retval < 0) {
      if (m_atm_logger) m_atm_logger->warn("[TChemATM] CVode returned error code " + std::to_string(retval));
    }
    
    // Copy results back from y to state
    Kokkos::parallel_for(
        "cvode_unpack_y", Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
        KOKKOS_LAMBDA(const int isample) {
          for (int j = 0; j < number_of_equations; ++j) {
            state(isample, j + 3) = y2d(isample, j);
          }
        });
    
  } else
#endif  // TCHEM_ATM_ENABLE_SUNDIALS
  {
    // ========================================================================
    // TChem Native Solvers Path (ImplicitEuler, TRBDF2, ExplicitEuler)
    // ========================================================================
    
    TChem::real_type tsum(0);
    for (int iter = 0; iter < m_max_time_iterations && tsum <= dt * 0.9999;
         ++iter) {

      if (m_solver_enum == SolverType::ImplicitEuler) {
        implicit_euler_type::runDeviceBatch(
           policy, m_tol_newton, m_tol_time, m_fac, tadv, m_state, m_photo_rates,
            m_external_sources, t_view, dt_view_local, m_state, m_workspace, m_kmcd);
      } else if (m_solver_enum == SolverType::TRBDF2) {
        trbdf2_type::runDeviceBatch(
            policy, m_tol_newton, m_tol_time, m_fac, tadv, m_state, m_photo_rates,
            m_external_sources, t_view, dt_view_local, m_state, m_kmcd);
      } else {
        explicit_euler_type::runDeviceBatch(
            policy, tadv, m_state, m_photo_rates, m_external_sources, t_view,
            dt_view_local, m_state, m_workspace, m_kmcd);
      }

      // Update time advance struct and compute average time for convergence check.
      // Note: parallel_reduce with a scalar reduction target implicitly fences
      // to return the result to the host, so no explicit fence is needed.
      tsum = 0;
      Kokkos::parallel_reduce(
          "tchem_update_tadv",
          Kokkos::RangePolicy<TChem::exec_space>(0, m_nsamples),
          KOKKOS_LAMBDA(const int i, TChem::real_type& update) {
            tadv(i)._tbeg = t_view(i);
            tadv(i)._dt   = dt_view_local(i);
            update += t_view(i);
          },
          tsum);
      tsum /= m_nsamples;
    }
  }

  // After the TChem run, convert dry-vmr state back to wet-mmr tracer fields.
  for (int ivar = 0; ivar < m_n_active_vars; ++ivar) {
    const auto& tracer_name = std::string(&m_species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    // Unpack only sampled entries from TChem state back into wet-mmr tracer field
    tchem::unpack_wet_mmr_from_state(q_tracer, state, qv, m_sample_icol, m_sample_ilev,
                  m_nsamples, ivar + 3, m_species_mw[ivar],
                  "tchem_copy_back_state_tracer");
  }

  //TODO:
  // run only w TChem-atm traces it looks like I also need mam4xx tracers.
  // Run tropopause
  // Run stratosphere
  // make a single test for case w aerosols.
  // get photolysis rates.
  // get external sources.
  // Future:
  // modify TChem-atm functions signature to pass tem and pressure
  // connect to aerosols.
  // use Analitycal Jacobian.
}

void TChemATM::finalize_impl() {
  // Clean up CVODE resources if allocated
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  if (m_cvode_mem != nullptr) {
    CVodeFree(&m_cvode_mem);
    m_cvode_mem = nullptr;
  }
  // Smart pointers will automatically clean up m_cvode_y, m_cvode_abstol, etc.
  m_sundials_ctx.reset();
  m_cvode_y.reset();
  m_cvode_abstol.reset();
  m_cvode_A.reset();
  m_cvode_LS.reset();
#endif
}

}  // namespace scream
