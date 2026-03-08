#include "eamxx_tchem_atm_process_interface.hpp"

#include <ekat_assert.hpp>

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

}  // namespace

TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;

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

  // Build TChem kinetic model metadata from the configured chemistry file.
  m_kmd = TChem::KineticModelData(chem_file);
  m_kmcd = TChem::createNCAR_KineticModelConstData<tchem_device_type>(m_kmd);
  m_tchem_ready = true;

  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int i = 0; i < m_kmd.nSpec_; ++i) {
    add_tracer<Updated>(std::string(&species_names_host(i, 0)), m_grid, q_unit);
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

  m_n_active_vars = m_kmcd.nSpec - m_kmcd.nConstSpec;
  m_state_vec_dim = TChem::Impl::getStateVectorSize(m_kmcd.nSpec);

  m_state = explicit_euler_type::real_type_2d_view_type("tchem_state", m_nbatch, m_state_vec_dim);
  m_photo_rates = explicit_euler_type::real_type_2d_view_type("tchem_photo_rates", m_nbatch, m_kmcd.nReac);
  m_external_sources = explicit_euler_type::real_type_2d_view_type("tchem_external_sources", m_nbatch, m_n_active_vars);
  m_t = explicit_euler_type::real_type_1d_view_type("tchem_time", m_nbatch);
  m_dt_view = explicit_euler_type::real_type_1d_view_type("tchem_dt", m_nbatch);
  m_tadv = TChem::time_advance_type_1d_view("tchem_tadv", m_nbatch);
}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::run_impl(const double dt) {
  EKAT_REQUIRE_MSG(m_tchem_ready,
                   "Error! TChemATM::run_impl called before TChem model initialization.\n");

  using ordinal_type = TChem::ordinal_type;

  if (m_nbatch == 0) {
    return;
  }

  const auto t_mid = get_field_in("T_mid").get_view<const Real **>();
  const auto p_mid = get_field_in("p_mid").get_view<const Real **>();
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

  // Populate TChem state pressure/temperature from EAMxx physics fields.
  fill_state_column_from_field(state, p_mid, nlevs, m_nbatch, 1,
                               "tchem_init_state_p");
  fill_state_column_from_field(state, t_mid, nlevs, m_nbatch, 2,
                               "tchem_init_state_t");

  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int ivar = 0; ivar < m_kmd.nSpec_; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    const auto& q_tracer = get_field_in(tracer_name).get_view< Real **>();
    //FIXME:
    //q_tracer from eamxx are mmr wet base, but TChem-atm expects vmr dry base. 
    // step 1. compute mmr dry base from mmr wet base and qv
    //const Real mmr_ispe_dry =
    //    PF::calculate_drymmr_from_wetmmr(mmr_igas(icol,kk), qv(icol, kk));
    // step 2. compute vmr dry base from mmr dry base
    //    state(ibacth, ispec) =  mam4::conversions::vmr_from_mmr(mmr_ispe_dry,mw_ispe);
    fill_state_column_from_field(state, q_tracer, nlevs, m_nbatch, ivar + 3,
                                 "tchem_init_state_tracer");
  }

  TChem::AtmosphericChemistryE3SM_ExplicitEuler::runDeviceBatch(
      policy, m_tadv, m_state, m_photo_rates, m_external_sources, m_t,
      m_dt_view, m_state,
      m_kmcd);

  // After the TChem run, copy the updated tracer values back to the EAMxx physics fields.
  for (int ivar = 0; ivar < m_kmd.nSpec_; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    fill_field_from_state_column(q_tracer, state, nlevs, m_nbatch, ivar + 3,
                                 "tchem_copy_back_state_tracer");
  } 
}

}  // namespace scream
