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
  using explicit_euler_type = TChem::AtmosphericChemistryE3SM_ExplicitEuler;

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
  TChem::ordinal_type m_n_active_vars = 0;
  TChem::ordinal_type m_state_vec_dim = 0;
  bool m_tchem_ready = false;
};

}  // namespace scream

#endif  // EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
