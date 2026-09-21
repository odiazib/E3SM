#ifndef EAMXX_TCHEM_ATM_CAMP_PROCESS_INTERFACE_HPP
#define EAMXX_TCHEM_ATM_CAMP_PROCESS_INTERFACE_HPP

#include "share/atm_process/atmosphere_process.hpp"
#include <ekat_parameter_list.hpp>
#include <ekat_kokkos_types.hpp>
#include <TChem.hpp>
#include <TChem_AerosolChemistry_ImplicitEuler.hpp>
#include <string>
#include <vector>

// CVODE batch solver support (guarded by Sundials availability)
#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
#include "TChem_AerosolChemistry_CVODE_RHS_Jacobian.hpp"
#endif

namespace scream {

class TChemATMCamp : public AtmosphereProcess {
 public:
  using tchem_device_type =
      typename Tines::UseThisDevice<TChem::exec_space>::type;

  // Type aliases for the two Tines-based solvers
  using implicit_euler_type = TChem::AerosolChemistry_ImplicitEuler;
  using trbdf2_type         = TChem::AerosolChemistry;

  // AerosolChemistry problem type (gas + aerosol, but we only use gas-phase
  // species for this CBO5 / CAMP interface).
  using problem_type =
      TChem::Impl::AerosolChemistry_Problem<Real, tchem_device_type>;

  using KT            = ekat::KokkosTypes<DefaultDevice>;
  using view_1d       = typename KT::template view_1d<Real>;
  using view_2d       = typename KT::template view_2d<Real>;
  using const_view_1d = typename KT::template view_1d<const Real>;
  using view_1d_int   = typename KT::template view_1d<int>;
  using host_view_1d  = Kokkos::View<Real*, Kokkos::HostSpace>;
  using ThreadTeam    = Kokkos::TeamPolicy<KT::ExeSpace>::member_type;

  using real_type_1d_view =
      Tines::value_type_1d_view<Real, tchem_device_type>;
  using real_type_2d_view =
      Tines::value_type_2d_view<Real, tchem_device_type>;
  using real_type_3d_view =
      Tines::value_type_3d_view<Real, tchem_device_type>;

#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  // CVODE Batch Solver Types
  using cvode_policy_type =
      typename TChem::UseThisTeamPolicy<TChem::exec_space>::type;
  using CVODEVecType  = sundials::kokkos::Vector<TChem::exec_space>;
  using CVODEMatType  = sundials::kokkos::DenseMatrix<TChem::exec_space>;
  using CVODELSType   = sundials::kokkos::DenseLinearSolver<TChem::exec_space>;
  using CVODESizeType = typename CVODEVecType::size_type;
#endif

  TChemATMCamp(const ekat::Comm& comm, const ekat::ParameterList& params);

  std::string name() const override { return "tchem_atm_camp"; }
  void create_requests() override;
  AtmosphereProcessType type() const override {
    return AtmosphereProcessType::Physics;
  }

  void initialize_impl(const RunType run_type) override;
  void run_impl(const double dt) override;
  void finalize_impl() override;

 private:
  std::shared_ptr<const AbstractGrid> m_grid;

  // Gas-phase kinetic model
  TChem::KineticModelData m_kmd;
  TChem::KineticModelNCAR_ConstData<tchem_device_type> m_kmcd;

  // Aerosol model data (required by AerosolChemistry_Problem even when only
  // gas-phase chemistry is desired; the aerosol mechanism file may contain
  // zero or more SIMPOL phase-transfer reactions).
  TChem::AerosolModelData m_amd;
  TChem::AerosolModel_ConstData<tchem_device_type> m_amcd;

  // Host mirror of species name strings
  decltype(std::declval<TChem::KineticModelData>().sNames_.view_host())
      m_species_names_host;

  // TChem state and working views
  real_type_2d_view m_state;
  real_type_2d_view m_num_concentration;  // particle number concentrations
  real_type_1d_view m_t;
  real_type_1d_view m_dt_view;
  TChem::time_advance_type_1d_view m_tadv;

  // Tolerance / scaling views for Tines implicit solvers
  real_type_1d_view m_tol_newton;
  real_type_2d_view m_tol_time;
  real_type_2d_view m_fac;

  // Species molecular weights indexed by TChem species order
  std::vector<Real> m_species_mw;

  int m_ncols            = 0;
  int m_nlevs            = 0;
  int m_nbatch           = 0;
  TChem::ordinal_type m_n_active_gas_vars = 0;
  TChem::ordinal_type m_n_active_vars     = 0;  // gas + aerosol active vars
  TChem::ordinal_type m_state_vec_dim     = 0;
  int m_total_n_species  = 0;
  bool m_tchem_ready     = false;

  // Number of constant species (M, N2, O2, H2O, H2, CH4 for CBO5).
  // Read from the mechanism automatically (kmcd.nConstSpec).
  int m_num_const_spec   = 0;

  // Constant-species indices in the full species array (not the state vector).
  // These are looked up by name from the mechanism file to avoid hard-coding
  // the constant-species ordering.
  int m_idx_M   = -1;
  int m_idx_N2  = -1;
  int m_idx_O2  = -1;
  int m_idx_H2O = -1;
  int m_idx_H2  = -1;
  int m_idx_CH4 = -1;

  // Solver selection
  enum class SolverType { CVODEBatch, ImplicitEuler, TRBDF2 };
  SolverType  m_solver_enum = SolverType::TRBDF2;
  std::string m_solver_type = "trbdf2";

  // Tines solver parameters (shared by ImplicitEuler and TRBDF2)
  int  m_max_time_iterations    = 1000;
  int  m_max_newton_iterations  = 100;
  int  m_jacobian_interval      = 1;
  Real m_dtmin_sub   = 1e-4;
  Real m_dtmax_sub   = -1.0;
  Real m_atol_newton = 1e-10;
  Real m_rtol_newton = 1e-6;
  Real m_atol_time   = 1e-12;
  Real m_rtol_time   = 1e-4;

  // CVODE parameters
  Real m_cvode_rtol      = 1e-4;
  Real m_cvode_atol      = 1e-12;
  int  m_cvode_max_steps = 10000;
  Real m_cvode_max_step  = -1.0;
  Real m_cvode_min_step  = 0.0;

  // Sampling views (all levels)
  view_1d_int m_sample_icol;
  view_1d_int m_sample_ilev;

#if defined(TCHEM_ATM_ENABLE_SUNDIALS)
  // CVODE batch solver state
  std::unique_ptr<sundials::Context> m_sundials_ctx;
  void* m_cvode_mem = nullptr;
  std::unique_ptr<CVODEVecType> m_cvode_y;
  std::unique_ptr<CVODEVecType> m_cvode_abstol;
  std::unique_ptr<sundials::ConvertibleTo<SUNMatrix>> m_cvode_A;
  std::unique_ptr<sundials::ConvertibleTo<SUNLinearSolver>> m_cvode_LS;

  TChem::UserData m_cvode_udata;
  real_type_1d_view m_cvode_temperature;
  real_type_1d_view m_cvode_pressure;
  real_type_2d_view m_cvode_const_tracers;
#endif
};

}  // namespace scream

#endif  // EAMXX_TCHEM_ATM_CAMP_PROCESS_INTERFACE_HPP
