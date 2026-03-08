#include "eamxx_tchem_atm_process_interface.hpp"

#include <ekat_assert.hpp>

namespace scream {

TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {
  using namespace ekat::units;
  constexpr auto q_unit = kg / kg;

  const auto m_grid = m_grids_manager->get_grid("physics");
  EKAT_REQUIRE_MSG(m_grid != nullptr,
                   "Error! TChemATM could not get 'physics' grid.\n");

  const auto chem_file = m_params.get<std::string>(
      "chem_file", m_params.get<std::string>("chemfile", ""));
  EKAT_REQUIRE_MSG(!chem_file.empty(),
                   "Error! Missing required parameter 'chem_file' for tchem_atm.\n");

  using device_type = typename Tines::UseThisDevice<TChem::exec_space>::type;

  // Build TChem kinetic model metadata from the configured chemistry file.
  TChem::KineticModelData kmd(chem_file);
  const auto kmcd = TChem::createNCAR_KineticModelConstData<device_type>(kmd);

  const auto species_names_host = kmd.sNames_.view_host();
  for (int i = 0; i < kmd.nSpec_; ++i) {
    add_tracer<Updated>(std::string(&species_names_host(i, 0)), m_grid, q_unit);
  }

  // Keep kmcd creation for upcoming tchem-atm integration work.
  // (void)kmcd;
}

void TChemATM::initialize_impl(const RunType /* run_type */) {}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::run_impl(const double /* dt */) {}

}  // namespace scream
