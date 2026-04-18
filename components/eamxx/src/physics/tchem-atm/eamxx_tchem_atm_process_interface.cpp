#include "eamxx_tchem_atm_process_interface.hpp"

#include <ekat_assert.hpp>
#include "share/physics/eamxx_common_physics_functions.hpp"

namespace scream {

namespace {

template <typename StateViewType, typename SourceViewType>
void fill_state_column_from_field(const StateViewType& state,
                                  const SourceViewType& src,
                                  const int nlevs,
                                  const int nbatch,
                                  const int state_col,
                                  const char* kernel_name) {
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        state(i, state_col) = src(icol, ilev);
      });
}

    template <typename StateViewType, typename TracerViewType, typename QvViewType>
    void fill_state_column_from_wet_mmr_field(const StateViewType& state,
                  const TracerViewType& q_tracer,
                  const QvViewType& qv,
                  const int nlevs,
                  const int nbatch,
                  const int state_col,
                  const Real species_mw,
                  const char* kernel_name) {
      using PF = scream::PhysicsFunctions<DefaultDevice>;
      constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;
      Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        const Real mmr_wet = q_tracer(icol, ilev);
        const Real qv_wet = qv(icol, ilev);
        const Real mmr_dry = PF::calculate_drymmr_from_wetmmr(mmr_wet, qv_wet);
        state(i, state_col) = mmr_dry * air_mw / species_mw;
      });
    }

    template <typename DestViewType, typename StateViewType>
    void fill_field_from_state_column(const DestViewType& dst,
                  const StateViewType& state,
                  const int nlevs,
                  const int nbatch,
                  const int state_col,
                  const char* kernel_name) {
      Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        dst(icol, ilev) = state(i, state_col);
      });
    }

template <typename DestViewType, typename StateViewType, typename QvViewType>
void fill_wet_mmr_field_from_state_column(const DestViewType& dst,
                                          const StateViewType& state,
                                          const QvViewType& qv,
                                          const int nlevs,
                                          const int nbatch,
                                          const int state_col,
                                          const Real species_mw,
                                          const char* kernel_name) {
  using PF = scream::PhysicsFunctions<DefaultDevice>;
  constexpr Real air_mw = scream::physics::Constants<Real>::MWdry.value;
  Kokkos::parallel_for(
      kernel_name, Kokkos::RangePolicy<TChem::exec_space>(0, nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        const Real qv_wet = qv(icol, ilev);
        const Real vmr_dry = state(i, state_col);
        const Real mmr_dry = vmr_dry * species_mw / air_mw;
        const Real qv_dry = PF::calculate_drymmr_from_wetmmr(qv_wet, qv_wet);
        dst(icol, ilev) = PF::calculate_wetmmr_from_drymmr(mmr_dry, qv_dry);
      });
}

}  // namespace

TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;
  // std::cout << "[TChemATM] create_requests\n";

  m_grid = m_grids_manager->get_grid("physics");
  EKAT_REQUIRE_MSG(m_grid != nullptr,
                   "Error! TChemATM could not get 'physics' grid.\n");

  const auto chem_file = m_params.get<std::string>(
      "chem_file", m_params.get<std::string>("chemfile", ""));
  EKAT_REQUIRE_MSG(!chem_file.empty(),
                   "Error! Missing required parameter 'chem_file' for tchem_atm.\n");

  const auto& grid_name = m_grid->name();
  const auto scalar3d_mid = m_grid->get_3d_scalar_layout(true);
  add_field<Required>("p_mid", scalar3d_mid, Pa, grid_name);
  add_field<Required>("T_mid", scalar3d_mid, K, grid_name);
  add_field<Required>("qv", scalar3d_mid, q_unit, grid_name);


  // Build TChem kinetic model metadata from the configured chemistry file.
  // std::cout << "[TChemATM] KineticModelData\n";
  m_kmd = TChem::KineticModelData(chem_file);
  // std::cout << "[TChemATM] createNCAR_KineticModelConstData\n";
  m_kmcd = TChem::createNCAR_KineticModelConstData<tchem_device_type>(m_kmd);
  // std::cout << "[TChemATM] Done KineticModelData "<<m_kmd.nSpec_ <<"\n";

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
    // std::cout << "[TChemATM] Molecular weight for species "<< i <<" " << sname << " = " << m_species_mw[i] << " g/mol\n";
  }
  // std::cout << "[TChemATM] Done loading molecular weights\n";
  m_tchem_ready = true;
   // std::cout << "[TChemATM] Done m_species_mw\n";

  //FIXME: invariants are not tracers.
  for (int i = 0; i < m_kmd.nSpec_ - m_num_invariants; ++i) {
    const std::string sname(&species_names_host(i, 0));
    // std::cout << "[TChemATM] species[" << i << "] = " << sname << "\n";
    add_tracer<Updated>(sname, m_grid, q_unit);
  }
  std::cout << "[TChemATM] Number of tracers added: " << m_kmd.nSpec_ - m_num_invariants << "\n";
    // Add prescribed constant tracer fields (oxidants).
  // M, N2, O2, H2O, H2, CH4 are computed from T and P at runtime, not registered as fields.
  constexpr int num_tracer_cnst = 3;
  for (int j = 0; j < num_tracer_cnst; ++j) {
    const std::string sname(&species_names_host(m_kmcd.M_index + 6 + j, 0));
    add_field<Updated>(sname, scalar3d_mid, q_unit, grid_name);
  }
  // std::cout << "[TChemATM] Done create_requests\n";
}

void TChemATM::initialize_impl(const RunType /* run_type */) {
  // std::cout << "[TChemATM] initialize_impl\n";
  EKAT_REQUIRE_MSG(m_tchem_ready,
                   "Error! TChemATM::initialize_impl called before TChem model initialization.\n");

  m_ncols = m_grid->get_num_local_dofs();
  m_nlevs = m_grid->get_num_vertical_levels();
  m_nbatch = m_ncols * m_nlevs;
  if (m_nbatch == 0) {
    return;
  }

  m_n_active_vars = m_kmcd.nSpec - m_kmcd.nConstSpec;
  m_state_vec_dim = TChem::Impl::getStateVectorSize(m_kmcd.nSpec);

  m_state = explicit_euler_type::real_type_2d_view_type("tchem_state", m_nbatch, m_state_vec_dim);
  const int m_photo_reactions= 22;
  m_photo_rates = explicit_euler_type::real_type_2d_view_type("tchem_photo_rates", m_nbatch, m_photo_reactions);
  m_external_sources = explicit_euler_type::real_type_2d_view_type("tchem_external_sources", m_nbatch, m_n_active_vars);
  m_t = explicit_euler_type::real_type_1d_view_type("tchem_time", m_nbatch);
  m_dt_view = explicit_euler_type::real_type_1d_view_type("tchem_dt", m_nbatch);
  m_tadv = TChem::time_advance_type_1d_view("tchem_tadv", m_nbatch);

  // Read solver/time-stepping parameters from the namelist.
  m_solver_type          = m_params.get<std::string>("solver_type", "implicit_euler");
  m_max_time_iterations  = m_params.get<int>("max_time_iterations", 100);
  m_jacobian_interval    = m_params.get<int>("jacobian_interval", 1);
  m_dtmin_sub            = m_params.get<double>("dtmin_sub", 1e-1);
  m_dtmax_sub            = m_params.get<double>("dtmax_sub", -1.0);
  m_atol_newton          = m_params.get<double>("atol_newton", 1e-10);
  m_rtol_newton          = m_params.get<double>("rtol_newton", 1e-6);
  m_atol_time            = m_params.get<double>("atol_time", 1e-12);
  m_rtol_time            = m_params.get<double>("rtol_time", 1e-4);
  m_use_shared_workspace = m_params.get<bool>("use_shared_workspace", true);
  std::cout << "[TChemATM] solver_type = " << m_solver_type << "\n";

  // Allocate and populate tolerance/scaling views for implicit solvers.
  if (m_solver_type == "implicit_euler" || m_solver_type == "trbdf2") {
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
    if (m_solver_type == "implicit_euler") {
      per_team_extent = TChem::AtmosphericChemistryE3SM::getWorkSpaceSize(m_kmcd);
    } else if (m_solver_type == "trbdf2") {
      per_team_extent = TChem::AtmosphericChemistryE3SM::getWorkSpaceSize(m_kmcd);
    } else {
      per_team_extent = TChem::AtmosphericChemistryE3SM_ExplicitEuler::getWorkSpaceSize(m_kmcd);
    }
    m_workspace = explicit_euler_type::real_type_2d_view_type(
        "tchem_workspace", m_nbatch, 10*per_team_extent);
  }
  // std::cout << "[TChemATM] Done initialize_impl\n";
}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::run_impl(const double dt) {
  

  // std::cout << "[TChemATM] run_impl with dt = " << dt << "\n";
  EKAT_REQUIRE_MSG(m_tchem_ready,
                   "Error! TChemATM::run_impl called before TChem model initialization.\n");
 
  using ordinal_type = TChem::ordinal_type;

  if (m_nbatch == 0) {
    return;
  }

  const auto& t_mid = get_field_in("T_mid").get_view<const Real **>();
  const auto& p_mid = get_field_in("p_mid").get_view<const Real **>();
  const auto& qv = get_field_in("qv").get_view<const Real **>();
  const int nlevs = m_nlevs;
  const auto state = m_state;

  using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;

  policy_type policy(TChem::exec_space(), m_nbatch, Kokkos::AUTO());
  ordinal_type per_team_extent = 0;
   if (m_solver_type == "implicit_euler") {
      per_team_extent = TChem::AtmosphericChemistryE3SM::getWorkSpaceSize(m_kmcd);
    } else if (m_solver_type == "trbdf2") {
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

  Kokkos::deep_copy(m_photo_rates, 0.0);
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
  tadv_default._max_num_newton_iterations = m_params.get<int>("max_newton_iterations", 100);
  tadv_default._num_time_iterations_per_interval = 100;
  tadv_default._jacobian_interval = m_jacobian_interval;
  Kokkos::deep_copy(m_tadv, tadv_default);

  // std::cout << "[TChemATM] Starting TChem run\n";

  // Populate TChem state pressure/temperature from EAMxx physics fields.
  fill_state_column_from_field(state, p_mid, nlevs, m_nbatch, 1,
                               "tchem_init_state_p");
  fill_state_column_from_field(state, t_mid, nlevs, m_nbatch, 2,
                               "tchem_init_state_t");
  // std::cout << "[TChemATM] Done fill_state_column_from_field\n";
  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int ivar = 0; ivar < m_kmd.nSpec_-m_num_invariants; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    // std::cout << "[TChemATM] Filling state column for tracer " << tracer_name << "\n";
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    
    fill_state_column_from_wet_mmr_field(state, q_tracer, qv, nlevs, m_nbatch,
                                         ivar + 3, m_species_mw[ivar],
                                         "tchem_init_state_tracer");
    // if (ivar == 0) {
    //   TChem::exec_space().fence();
    //   auto state_host = Kokkos::create_mirror_view_and_copy(
    //       Kokkos::HostSpace(), state);
    //   std::cout << "[TChemATM] O3 VMR col 0 (before TChem):";
    //   for (int ilev = 0; ilev < state_host.extent(0); ++ilev)
    //     std::cout << " " << state_host(ilev, ivar + 3);
    //   std::cout << "\n";
    // }
  }
  // std::cout << "[TChemATM] Done fill_state_column_from_wet_mmr_field\n";
 #if 1
     // conversion factor for Pascals to dyne/cm^2
  constexpr Real Pa_xfac = 10.0;
  // presumably, the boltzmann constant, in CGS units
  constexpr Real boltz_cgs = 0.13806500000000001E-015;
  // Compute invariants:
  // step 1: M [molecules/cm^3] = Pa_xfac * P [Pa] / (boltz_cgs * T [K])
  const int m_state_col = m_kmcd.M_index + 3;
  constexpr int num_tracer_cnst = 3;
  Kokkos::parallel_for(
      "tchem_compute_M", Kokkos::RangePolicy<TChem::exec_space>(0, m_nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        const Real m_value =
            Pa_xfac * p_mid(icol, ilev) / (boltz_cgs * t_mid(icol, ilev));
        state(i, m_state_col) = m_value;
        // N2 = 0.79 * M
        state(i, m_state_col + 1) = 0.79;// * m_value;
        // O2 = 0.21 * M
        state(i, m_state_col + 2) = 0.21;// * m_value;
        // H2O = qv * M / (1 + qv)
        state(i, m_state_col + 3) = qv(icol, ilev) / (1.0 + qv(icol, ilev));//m_value;
        // H2 = 5.5e-7 * M
        state(i, m_state_col + 4) = 5.5e-7;// * m_value;
        // CH4 =0;
        state(i, m_state_col + 5) = 0.0;
        
      });
 

  for (int j = 0; j < num_tracer_cnst; ++j) {
    const auto& tracer_name = std::string(&species_names_host(m_kmcd.M_index + 6 + j, 0));
    // std::cout << "[TChemATM] Filling state column for invariant tracer " << tracer_name << "\n";
    const auto& q_tracer = get_field_out(tracer_name).get_view<Real **>();
    const int state_col_j = m_state_col + 6 + j;
    Kokkos::parallel_for(
      "tchem_compute_cnst_tracer", Kokkos::RangePolicy<TChem::exec_space>(0, m_nbatch),
      KOKKOS_LAMBDA(const int i) {
        const int icol = i / nlevs;
        const int ilev = i % nlevs;
        state(i, state_col_j) = q_tracer(icol, ilev);// * state(i, m_state_col);
      });
  }
  // Time loop: mirrors TChem_AtmosphericChemistryE3SM.cpp standalone example.
  // Solver type and time-stepping parameters are controlled via namelist.
  TChem::real_type tsum(0);
  const auto& tadv        = m_tadv;
  const auto& t_view      = m_t;
  const auto& dt_view     = m_dt_view;

  for (int iter = 0; iter < m_max_time_iterations && tsum <= dt * 0.9999;
       ++iter) {
#if 1
    if (m_solver_type == "implicit_euler") {
      implicit_euler_type::runDeviceBatch(
          policy, m_tol_newton, m_tol_time, m_fac, tadv, m_state, m_photo_rates,
          m_external_sources, t_view, dt_view, m_state, m_workspace, m_kmcd);
    } else if (m_solver_type == "trbdf2") {
      trbdf2_type::runDeviceBatch(
          policy, m_tol_newton, m_tol_time, m_fac, tadv, m_state, m_photo_rates,
          m_external_sources, t_view, dt_view, m_state, m_kmcd);
    } else {
      explicit_euler_type::runDeviceBatch(
          policy, tadv, m_state, m_photo_rates, m_external_sources, t_view,
          dt_view, m_state, m_workspace, m_kmcd);
    }
    TChem::exec_space().fence();
    tsum = 0;
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<TChem::exec_space>(0, m_nbatch),
        KOKKOS_LAMBDA(const int i, TChem::real_type& update) {
          tadv(i)._tbeg = t_view(i);
          tadv(i)._dt   = dt_view(i);
          update += t_view(i);
        },
        tsum);
    Kokkos::fence();
    tsum /= m_nbatch;
  }
#endif
  // Print O3 VMR for column 0 after the TChem run.
  // for (int ivar = 0; ivar < m_kmcd.nSpec - m_kmcd.nConstSpec; ++ivar) {
  //   const std::string sname(&species_names_host(ivar, 0));
  //   if (ivar == 0) {
  //     TChem::exec_space().fence();
  //     auto m_state_host = Kokkos::create_mirror_view_and_copy(
  //         Kokkos::HostSpace(), m_state);
  //     std::cout << "[TChemATM] O3 VMR col 0 (after TChem):";
  //     for (int ilev = 0; ilev < nlevs; ++ilev)
  //       std::cout << " " << m_state_host(ilev, ivar + 3);
  //     std::cout << "\n";
  //     break;
  //   }
  // }

  // After the TChem run, convert dry-vmr state back to wet-mmr tracer fields.
  for (int ivar = 0; ivar < m_kmcd.nSpec - m_kmcd.nConstSpec; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    fill_wet_mmr_field_from_state_column(q_tracer, state, qv, nlevs, m_nbatch,
                                         ivar + 3, m_species_mw[ivar],
                                         "tchem_copy_back_state_tracer");
  } 

  //TODO:
  // run only w TChem-atm traces it looks like I also need mam4xx tracers. 
  // Run tropopause 
  // Run stratoshere
  // get num_tracer_cnst
  // make a single test for case w aerosols.
  // get photolysis rates.
  // get external sources.
  // Future:
  // modify TChem-atm functions signature to pass tem and pressure
  // connect to aerosols. 
  // use Analitycal Jacobian.
#endif  
}

}  // namespace scream
