#include "eamxx_tchem_atm_process_interface.hpp"
#include "eamxx_tchem_atm_tchem_functions.hpp"

#include <ekat_assert.hpp>
#include <ekat_team_policy_utils.hpp>
#include <mam4xx/mam4.hpp>
#include <mam4xx/mo_photo.hpp>

#include "physics/rrtmgp/shr_orb_mod_c2f.hpp"
#include "share/physics/eamxx_common_physics_functions.hpp"

namespace scream {


TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;
  using namespace ShortFieldTagsNames;
  // std::cout << "[TChemATM] create_requests\n";

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

  // Read sampling configuration: sample above tropopause (true) or below (false)
  m_run_troposhere = m_params.get<bool>("m_run_troposhere", true);
  if (m_atm_logger) m_atm_logger->info("[TChemATM] m_run_troposhere = " + std::to_string(m_run_troposhere));

  //FIXME: invariants are not tracers.
  for (int i = 0; i < m_kmd.nSpec_ - m_num_invariants; ++i) {
    const std::string sname(&species_names_host(i, 0));
    // std::cout << "[TChemATM] species[" << i << "] = " << sname << "\n";
    add_tracer<Updated>(sname, m_grid, q_unit);
  }
  if (m_atm_logger) m_atm_logger->info("[TChemATM] Number of tracers added: " + std::to_string(m_kmd.nSpec_ - m_num_invariants));
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

  // Match MAM behavior: cache direct visible surface albedo view once.
  m_sfc_alb_dir_vis = get_field_in("sfc_alb_dir_vis").get_view<const Real *>();

  m_n_active_vars = m_kmcd.nSpec - m_kmcd.nConstSpec;
  m_state_vec_dim = TChem::Impl::getStateVectorSize(m_kmcd.nSpec);

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
  m_ilev_tropp = view_1d_int("tchem_ilev_tropp", m_ncols);
  // Allocate persistent index/offset views once here and reuse in run_impl.
  m_offsets = view_1d_int("tchem_offsets", m_ncols + 1);
  m_sample_icol = view_1d_int("tchem_sample_icol", m_nbatch);
  m_sample_ilev = view_1d_int("tchem_sample_ilev", m_nbatch);
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
  if (m_atm_logger) m_atm_logger->info("[TChemATM] solver_type = " + m_solver_type);

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
  // Photo table initialization (optional)
  const std::string rsf_file = m_params.get<std::string>("mam4_rsf_file", "");
  const std::string xs_long_file = m_params.get<std::string>("mam4_xs_long_file", "");
  if (!rsf_file.empty() && !xs_long_file.empty()) {
    m_photo_table = tchem::read_photo_table_uci(rsf_file, xs_long_file);
    m_photo_table_len = mam4::mo_photo::get_photo_table_work_len(m_photo_table);
    m_work_photo_table = view_2d("tchem_photo_work", m_ncols, m_photo_table_len);
    // allocate a 3D photo buffer: (ncols, pver, phtcnt)
    m_photo_3d = view_3d("tchem_photo_3d", m_ncols, mam4::mo_photo::pver, mam4::mo_photo::phtcnt);
    // allocate O3 column buffer
    m_o3col = view_2d("tchem_o3col", m_ncols, mam4::mo_photo::pver);
    // find O3 species index in kinetic model names (if present)
    m_o3_species_index = -1;
    auto species_names_host = m_kmd.sNames_.view_host();
    for (int i = 0; i < m_kmd.nSpec_; ++i) {
      const std::string sname(&species_names_host(i, 0));
      if (sname == "O3") { m_o3_species_index = i; break; }
    }
    m_have_photo_table = true;
  } else {
    m_have_photo_table = false;
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

  // Compute interface geopotential heights
  Kokkos::parallel_for(
    "tchem_z_int", col_policy,
    KOKKOS_LAMBDA(const ThreadTeam& team) {
      const int icol = team.league_rank();
      PF::calculate_z_int(team, nlevs, ekat::subview(dz, icol),
                          z_surf, ekat::subview(z_iface, icol));
    });

  // Compute midpoint geopotential heights
  Kokkos::parallel_for(
    "tchem_z_mid", col_policy,
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

  // Choose sampling option: sample above or below tropopause according to
  // the 'm_run_troposhere' parameter (true = above, false = below).
  const bool above = m_run_troposhere;
  m_nsamples = tchem::compute_nsamples(ntropopause, ncol, nlev, above);
  if (m_atm_logger) m_atm_logger->info("[TChemATM] m_nsamples = " + std::to_string(m_nsamples));

  using policy_type = typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
 
  policy_type policy(TChem::exec_space(), m_nsamples, Kokkos::AUTO());
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
  // Compute photo table rates if we have a photo table
  if (m_have_photo_table) {
    // Compute orbital eccentricity factor used by MAM photo_table.
    int orbital_year = m_params.get<int>("orbital_year", -9999);
    double eccen = m_params.get<double>("orbital_eccentricity", -9999.0);
    double obliq = m_params.get<double>("orbital_obliquity", -9999.0);
    double mvelp = m_params.get<double>("orbital_mvelp", -9999.0);
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

    // Match MAM behavior: compute zenith angle on host, then copy to device.
    auto zenith_host = Kokkos::create_mirror_view(m_zenith_angle);
    const auto col_latitudes_host =
      m_grid->get_geometry_data("lat").get_view<const Real *, Host>();
    const auto col_longitudes_host =
      m_grid->get_geometry_data("lon").get_view<const Real *, Host>();
    for (int i = 0; i < m_ncols; ++i) {
      const Real lat = col_latitudes_host(i) * M_PI / 180.0;
      const Real lon = col_longitudes_host(i) * M_PI / 180.0;
      const Real cosz = shr_orb_cosz_c2f(calday, lat, lon, delta, dt);
      zenith_host(i) = acos(cosz);
    }
    Kokkos::deep_copy(m_zenith_angle, zenith_host);

    // zero the 3D photo buffer
    Kokkos::deep_copy(m_photo_3d, 0.0);
    // Prepare inputs
    const auto& sfc_alb = m_sfc_alb_dir_vis;
    const auto& zenith_angle = m_zenith_angle;
    const auto& qc_field = get_field_in("qc").get_view<const Real **>();
    const auto& cldfrac = get_field_in("cldfrac_tot").get_view<const Real **>();

    view_2d o3_field;
    bool have_o3_field = false;
    if (m_o3_species_index >= 0) {
      const auto species_names_host = m_kmd.sNames_.view_host();
      const std::string o3_name(&species_names_host(m_o3_species_index, 0));
      o3_field = get_field_out(o3_name).get_view<Real **>();
      have_o3_field = true;
    }

    // ozone column buffer (preallocated in initialize_impl)
    Kokkos::deep_copy(m_o3col, 0.0);
#if 1
    Kokkos::parallel_for(
        "tchem_photo_table", col_policy,
        KOKKOS_LAMBDA(const ThreadTeam &team) {
          const int icol = team.league_rank();
          // per-column work array (1D view)
          const auto work_photo_table_icol = ekat::subview(m_work_photo_table, icol);
          mam4::mo_photo::PhotoTableWorkArrays photo_work_arrays_icol;
          mam4::mo_photo::set_photo_table_work_arrays(m_photo_table, work_photo_table_icol,
                                                     photo_work_arrays_icol);
          team.team_barrier();
          // subviews for column inputs
          const auto pmid_col = ekat::subview(p_mid, icol);
          const auto pdel_col = ekat::subview(p_del_dry, icol);
          const auto t_col = ekat::subview(t_mid, icol);
          const auto o3_col = ekat::subview(m_o3col, icol);
          // compute o3 column densities if we have an O3 tracer
          if (have_o3_field) {
            const auto mmr_o3_col = ekat::subview(o3_field, icol);
            // compute column densities (molecules/cm^2) from mmr and pdel
            mam4::microphysics::compute_o3_column_density(team, pdel_col, mmr_o3_col,
                                                          0.0, m_species_mw[m_o3_species_index],
                                                          o3_col);
          }
          const Real srfalb = sfc_alb(icol);
          const auto qc_col = ekat::subview(qc_field, icol);
          const auto cld_col = ekat::subview(cldfrac, icol);
          const auto photo_icol = ekat::subview(m_photo_3d, icol);
          const Real esfact = eccf;
          mam4::mo_photo::table_photo(team, photo_icol, pmid_col, pdel_col, t_col,
                                     o3_col, zenith_angle(icol), srfalb, qc_col, cld_col,
                                     esfact, m_photo_table, photo_work_arrays_icol);
        });

    // copy into m_photo_rates (flattened nbatch x nphoto_reactions)
    const int phtcnt = mam4::mo_photo::phtcnt;
    Kokkos::parallel_for(
        "tchem_photo_copy", Kokkos::RangePolicy<TChem::exec_space>(0, m_ncols),
        KOKKOS_LAMBDA(const int icol) {
          for (int kk = 0; kk < m_nlevs; ++kk) {
            const int ibatch = icol * m_nlevs + kk;
            for (int mm = 0; mm < phtcnt && mm < m_photo_rates.extent(1); ++mm) {
              m_photo_rates(ibatch, mm) = m_photo_3d(icol, kk, mm);
            }
          }
        });
    Kokkos::fence();
#endif    
  }


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

  // compute offsets into the pre-allocated offsets view
  tchem::compute_offsets(ntropopause, ncol, nlev, m_offsets, above);
  // fill sample index arrays into the pre-allocated views
  tchem::compute_sample_indices(ntropopause, m_offsets, ncol, nlev, m_sample_icol,
                         m_sample_ilev, above);
                         
  // print m_offsets for debugging
  auto offsets_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_offsets);
  auto ntropopause_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ntropopause);
  // std::cout << "[TChemATM] Offsets: ";
  // for (int i = 0; i <= ncol; ++i)
    // std::cout << "offsets_host (" << i << ") = " << offsets_host(i) << ", ntropopause_host (" << i << ") = " << ntropopause_host(i) << "\n  ";
  // std::cout << "\n";  

  // auto sample_icol_host =
      // Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_sample_icol);
  // auto sample_ilev_host =
      // Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), m_sample_ilev);
  // std::cout << "[TChemATM] Sample indices (isample, icol, ilev):\n";
  // for (int isample = 0; isample < m_nsamples; ++isample) {
    // std::cout << "  (" << isample << ", " << sample_icol_host(isample)
              // << ", " << sample_ilev_host(isample) << ")\n";
  // }

  // std::cout << "  (" << 483 << ", " << sample_icol_host(483)
  //             << ", " << sample_ilev_host(483) << ")\n";
  // for (int isample = 0; isample < m_nsamples; ++isample) {
  //   if (sample_ilev_host(isample) < ntropopause_host(sample_icol_host(isample))) {
  //     std::cout << "  stratosphere sample: (" << isample << ", " << sample_icol_host(isample)
  //               << ", " << sample_ilev_host(isample) << ")\n";
  //   }
  // }  



  Kokkos::fence();

  // Pack pressure and temperature into the state for all selected samples
  tchem::pack_into_state(state, p_mid, m_sample_icol, m_sample_ilev, m_nsamples, 1,
                  "tchem_init_state_p");
  tchem::pack_into_state(state, t_mid, m_sample_icol, m_sample_ilev, m_nsamples, 2,
                  "tchem_init_state_t");

  // std::cout << "[TChemATM] Done fill_state_column_from_field\n";
  const auto species_names_host = m_kmd.sNames_.view_host();
  for (int ivar = 0; ivar < m_kmd.nSpec_-m_num_invariants; ++ivar) {
    const auto& tracer_name = std::string(&species_names_host(ivar, 0));
    // std::cout << "[TChemATM] Filling state column for tracer " << tracer_name << "\n";
    const auto& q_tracer = get_field_out(tracer_name).get_view< Real **>();
    // Use sampling-aware pack to only pack selected samples (m_sample_icol/ilev)
    tchem::pack_wet_mmr_into_state(state, q_tracer, qv, m_sample_icol, m_sample_ilev,
                m_nsamples, ivar + 3, m_species_mw[ivar],
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
    // Unpack only sampled entries from TChem state back into wet-mmr tracer field
    tchem::unpack_wet_mmr_from_state(q_tracer, state, qv, m_sample_icol, m_sample_ilev,
                  m_nsamples, ivar + 3, m_species_mw[ivar],
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
