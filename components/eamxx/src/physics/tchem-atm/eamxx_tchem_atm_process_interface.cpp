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
  std::cout << "[TChemATM] create_requests\n";

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
  std::cout << "[TChemATM] KineticModelData\n";
  m_kmd = TChem::KineticModelData(chem_file);
  std::cout << "[TChemATM] createNCAR_KineticModelConstData\n";
  m_kmcd = TChem::createNCAR_KineticModelConstData<tchem_device_type>(m_kmd);
  std::cout << "[TChemATM] Done KineticModelData "<<m_kmd.nSpec_ <<"\n";
  // TODO:add MQ in the init file.
  // m_species_mw = m_params.get<std::vector<Real>>("species_mw");
  // EKAT_REQUIRE_MSG(static_cast<int>(m_species_mw.size()) == m_kmd.nSpec_,
                  //  "Error! Parameter 'species_mw' must have one molecular weight per TChem species.\n");
   
   // TODO: initialze m_species_mw with ones. 
   m_species_mw = std::vector<Real>(m_kmd.nSpec_, 1.0);
   m_tchem_ready = true;
   std::cout << "[TChemATM] Done m_species_mw\n";

  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int i = 0; i < m_kmd.nSpec_; ++i) {
    const std::string sname(&species_names_host(i, 0));
    std::cout << "[TChemATM] species[" << i << "] = " << sname << "\n";
    add_tracer<Updated>(sname, m_grid, q_unit);
  }
  std::cout << "[TChemATM] Done create_requests\n";
}

void TChemATM::initialize_impl(const RunType /* run_type */) {
  std::cout << "[TChemATM] initialize_impl\n";
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
  m_photo_rates = explicit_euler_type::real_type_2d_view_type("tchem_photo_rates", m_nbatch, m_kmcd.nReac);
  m_external_sources = explicit_euler_type::real_type_2d_view_type("tchem_external_sources", m_nbatch, m_n_active_vars);
  m_t = explicit_euler_type::real_type_1d_view_type("tchem_time", m_nbatch);
  m_dt_view = explicit_euler_type::real_type_1d_view_type("tchem_dt", m_nbatch);
  m_tadv = TChem::time_advance_type_1d_view("tchem_tadv", m_nbatch);
  std::cout << "[TChemATM] Done initialize_impl\n";
}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::run_impl(const double dt) {
  std::cout << "[TChemATM] run_impl with dt = " << dt << "\n";
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
  const ordinal_type per_team_extent = explicit_euler_type::getWorkSpaceSize(m_kmcd);
  const ordinal_type per_team_scratch =
      TChem::Scratch<TChem::real_type_1d_view>::shmem_size(per_team_extent);
  policy.set_scratch_size(1, Kokkos::PerTeam(per_team_scratch));

  Kokkos::deep_copy(m_photo_rates, 0.0);
  Kokkos::deep_copy(m_external_sources, 0.0);
  Kokkos::deep_copy(m_t, 0.0);
  Kokkos::deep_copy(m_dt_view, dt);

  TChem::time_advance_type tadv_default;
  tadv_default._tbeg = 0;
  tadv_default._tend = dt;
  tadv_default._dt = dt;
  tadv_default._dtmin = dt;
  tadv_default._dtmax = dt;
  tadv_default._max_num_newton_iterations = 1;
  tadv_default._num_time_iterations_per_interval = 1;
  tadv_default._jacobian_interval = 1;
  Kokkos::deep_copy(m_tadv, tadv_default);

  std::cout << "[TChemATM] Starting TChem run\n";

  // Populate TChem state pressure/temperature from EAMxx physics fields.
  fill_state_column_from_field(state, p_mid, nlevs, m_nbatch, 1,
                               "tchem_init_state_p");
  fill_state_column_from_field(state, t_mid, nlevs, m_nbatch, 2,
                               "tchem_init_state_t");
  std::cout << "[TChemATM] Done fill_state_column_from_field\n";
  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int ivar = 0; ivar < m_kmd.nSpec_; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    std::cout << "[TChemATM] Filling state column for tracer " << tracer_name << "\n";
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    
    fill_state_column_from_wet_mmr_field(state, q_tracer, qv, nlevs, m_nbatch,
                                         ivar + 3, m_species_mw[ivar],
                                         "tchem_init_state_tracer");
  }
  std::cout << "[TChemATM] Done fill_state_column_from_wet_mmr_field\n";

  TChem::AtmosphericChemistryE3SM_ExplicitEuler::runDeviceBatch(
      policy, m_tadv, m_state, m_photo_rates, m_external_sources, m_t,
      m_dt_view, m_state,
      m_kmcd);

  // After the TChem run, convert dry-vmr state back to wet-mmr tracer fields.
  for (int ivar = 0; ivar < m_kmd.nSpec_; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    fill_wet_mmr_field_from_state_column(q_tracer, state, qv, nlevs, m_nbatch,
                                         ivar + 3, m_species_mw[ivar],
                                         "tchem_copy_back_state_tracer");
  } 

  //TODO:
  // modify TChem-atm functions signature to pass tem and pressure
  // get mw from uci yaml file. 
  // run test with traces.
  // Future:
  // connect to aerosols. 
  
}

}  // namespace scream
