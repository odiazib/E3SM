#ifndef EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
#define EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
#include "share/atm_process/atmosphere_process.hpp"
#include <ekat_parameter_list.hpp>
#include <TChem.hpp>
#include <string>
#include <vector>

namespace scream {

class TChemATM : public AtmosphereProcess {
 public:
  using tchem_device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;
  using explicit_euler_type  = TChem::AtmosphericChemistryE3SM_ExplicitEuler;
  using implicit_euler_type  = TChem::AtmosphericChemistryE3SM_ImplicitEuler;
  using trbdf2_type          = TChem::AtmosphericChemistryE3SM;

  TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params);

  std::string name() const override { return "tchem_atm"; }
  void create_requests() override;
  AtmosphereProcessType type() const override {
    return AtmosphereProcessType::Physics;
  }

//  protected:
  void initialize_impl(const RunType run_type) override;
  void run_impl(const double dt) override;
  void finalize_impl() override {}

 private:
  int get_len_temporary_views();
  void init_temporary_views();

  std::shared_ptr<const AbstractGrid> m_grid;
  TChem::KineticModelData m_kmd;
  TChem::KineticModelNCAR_ConstData<tchem_device_type> m_kmcd;
  explicit_euler_type::real_type_2d_view_type m_state;
  explicit_euler_type::real_type_2d_view_type m_photo_rates;
  explicit_euler_type::real_type_2d_view_type m_external_sources;
  explicit_euler_type::real_type_1d_view_type m_t;
  explicit_euler_type::real_type_1d_view_type m_dt_view;
  TChem::time_advance_type_1d_view m_tadv;
  std::vector<Real> m_species_mw;
  int m_ncols = 0;
  int m_nlevs = 0;
  int m_nbatch = 0;
  int m_num_invariants=0;
  TChem::ordinal_type m_n_active_vars = 0;
  TChem::ordinal_type m_state_vec_dim = 0;
  bool m_tchem_ready = false;
  // Solver selection and time-stepping parameters (read from namelist).
  std::string m_solver_type = "explicit_euler";
  int m_max_time_iterations = 1000;
  int m_jacobian_interval   = 1;
  Real m_dtmin_sub   = 1e-4;
  Real m_dtmax_sub   = -1.0; // negative → use physics dt
  Real m_atol_newton = 1e-10;
  Real m_rtol_newton = 1e-6;
  Real m_atol_time   = 1e-12;
  Real m_rtol_time   = 1e-4;
  // Tolerance and scaling views used by implicit solvers.
  explicit_euler_type::real_type_1d_view_type m_tol_newton;
  explicit_euler_type::real_type_2d_view_type m_tol_time;
  explicit_euler_type::real_type_2d_view_type m_fac;
  // Pre-allocated global workspace [nbatch x per_team_extent].
  // Used when m_use_shared_workspace is false.
  explicit_euler_type::real_type_2d_view_type m_workspace;
  bool m_use_shared_workspace = true;
};

}  // namespace scream

#endif  // EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
