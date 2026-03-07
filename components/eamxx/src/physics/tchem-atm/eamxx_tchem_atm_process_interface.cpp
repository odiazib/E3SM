#include "physics/tchem-atm/eamxx_tchem_atm_process_interface.hpp"

namespace scream {

TChemATM::TChemATM(const ekat::Comm& comm, const ekat::ParameterList& params)
    : AtmosphereProcess(comm, params) {}

void TChemATM::create_requests() {}

void TChemATM::initialize_impl(const RunType /* run_type */) {}

int TChemATM::get_len_temporary_views() {
  return 0;
}

void TChemATM::init_temporary_views() {}

void TChemATM::run_impl(const double /* dt */) {}

}  // namespace scream
