#ifndef EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
#define EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
#include "share/atm_process/atmosphere_process.hpp"
#include <ekat_parameter_list.hpp>
#include <TChem.hpp>
#include <string>

namespace scream {

class TChemATM : public AtmosphereProcess {
 public:
  TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params);

  std::string name() const override { return "tchem_atm"; }
  void create_requests() override;
  AtmosphereProcessType type() const override {
    return AtmosphereProcessType::Physics;
  }

 protected:
  void initialize_impl(const RunType run_type) override;
  void run_impl(const double dt) override;
  void finalize_impl() override {}

 private:
  int get_len_temporary_views();
  void init_temporary_views();
};

}  // namespace scream

#endif  // EAMXX_TCHEM_ATM_PROCESS_INTERFACE_HPP
