#include "eamxx_tchem_atm_tchem_functions.hpp"

#include <mam4xx/mam4.hpp>

#include <vector>

namespace scream {
namespace impl {

using HostViewInt1D = mam4::DeviceType::view_1d<int>::host_mirror_type;

mam4::mo_photo::PhotoTableData read_photo_table(
    const std::string& rsf_file, const std::string& xs_long_file,
    const std::vector<std::string>& rxt_names, int numj,
    const HostViewInt1D& lng_indexer_h);

}  // namespace impl

namespace tchem {

namespace {
using HostView1D    = mam4::DeviceType::view_1d<Real>::host_mirror_type;

void modify_photo_table_pht_alias_mult_1(mam4::mo_photo::PhotoTableData& table_data) {
  constexpr int phtcnt = mam4::mo_photo::phtcnt;
  auto pht_alias_mult_host = Kokkos::create_mirror_view(table_data.pht_alias_mult_1);

  for (int i = 0; i < phtcnt; ++i) {
    pht_alias_mult_host(i) = 1.0;
  }
  pht_alias_mult_host(19) = 0.0004;
  pht_alias_mult_host(20) = 0.0004;
  pht_alias_mult_host(21) = 0.0004;

  Kokkos::deep_copy(table_data.pht_alias_mult_1, pht_alias_mult_host);
}


void modify_photo_table_etfphot_e3smv3(mam4::mo_photo::PhotoTableData& table_data) {
    // We obtained these values from an e3sm simulations.
  // We should only use this function on Host.
  // Note: we need to review why this values are diferent in v2 and v3. 
  std::vector<Real> etfphot_data = {
       0.74420021091699609E+012,  0.85139533174377686E+012,  0.10185394901813281E+013,  0.11650271747566599E+013,  0.21278360656789287E+013,
       0.32102726730546094E+013,  0.36991317530087681E+013,  0.43854806251147578E+013,  0.46490589507128838E+013,  0.60824175448989688E+013,  
       0.45199732440082490E+013,  0.53096336119713838E+013,  0.46664963443549541E+013,  0.53889164471417969E+013,  0.44658315309513691E+013,  
       0.68485884416423164E+013,  0.61529859957264326E+013,  0.60917831402007217E+013,  0.57317396272234600E+013,  0.76284401653460156E+013,  
       0.13914324350439311E+014,  0.12044961828579564E+014,  0.28534210610292191E+014,  0.32105034161825109E+014,  0.24909006558383281E+014,  
       0.27759500542452164E+014,  0.23169477753649539E+014,  0.36246132305883531E+014,  0.61726886085997141E+014,  0.77952511543866469E+014, 
       0.76352487161941375E+014,  0.76176465839742719E+014,  0.94541477991598359E+014,  0.10114511241674064E+015,  0.10344934625369592E+015,  
       0.10990672653798267E+015,  0.10881184173185175E+015,  0.11372916232957753E+015,  0.13482418576882047E+015,  0.15934045446957247E+015,  
       0.14973282281420931E+015,  0.15173151415232916E+015,  0.15979204811229056E+015,  0.16966298325017691E+015,  0.18761242559642497E+015,  
       0.16419159493501566E+015,  0.18360595355117650E+015,  0.21957036295763366E+015,  0.19601524134628047E+015,  0.22383825011292891E+015,  
       0.18401883147059519E+015,  0.20100016593284484E+015,  0.20520669368043381E+015,  0.24319726136783316E+015,  0.35073290816277088E+015,  
       0.34511498344381438E+015,  0.35743855330938131E+015,  0.36617357240278300E+015,  0.34963637002805588E+015,  0.35554865920003012E+015,  
       0.42823524623318300E+015,  0.48406971868781850E+015,  0.49507991937358400E+015,  0.52693121969187562E+015,  0.52397774332888600E+015,  
       0.50873156915913800E+015,  0.48775293858117969E+015};
  auto etfphot_h    = HostView1D((Real *)etfphot_data.data(), table_data.nw);
  Kokkos::deep_copy(table_data.etfphot, etfphot_h);
}

}  // namespace

mam4::mo_photo::PhotoTableData read_photo_table_uci(
    const std::string& rsf_file, const std::string& xs_long_file) {
  using HostViewInt1D = mam4::DeviceType::view_1d<int>::host_mirror_type;
  const int phtcnt = mam4::mo_photo::phtcnt;
  HostViewInt1D lng_indexer_h("lng_indexer", phtcnt);

  std::vector<std::string> rxt_names = {
      "jo1dU",   "jo2_b",      "jh2o2",     "jch2o_a",   "jch2o_b",
      "jch3ooh", "jc2h5ooh",   "jno2",      "jno3_a",    "jno3_b",
      "jn2o5_a", "jn2o5_b",    "jhno3",     "jho2no2_a", "jho2no2_b",
      "jch3cho", "jpan",       "jacet",     "jmvk",      "jsoa_a1",
      "jsoa_a2", "jsoa_a3"};
  const std::vector<std::string> pht_alias_lst_2 = {
      "jo3_a",   // 0
      "NONE",    // 1
      "NONE",    // 2
      "NONE",    // 3
      "NONE",    // 4
      "NONE",    // 5
      "jch3ooh", // 6
      "NONE",    // 7
      "NONE",    // 8
      "NONE",    // 9
      "NONE",    // 10
      "NONE",    // 11
      "NONE",    // 12
      "NONE",    // 13
      "NONE",    // 14
      "NONE",    // 15
      "NONE",    // 16
      "NONE",    // 17
      "NONE",    // 18
      "jno2",    // 19
      "jno2",    // 20
      "jno2"     // 21
  };

  std::vector<int> photo_inti = {1,  2,  3, 4, 5, 6, 6, 7,  8, 9, 10,
                                 11, 12, 13, 14, 15, 16, 17, 18, 7, 7, 7};
  for (int i = 0; i < phtcnt; ++i) {
    lng_indexer_h(i) = photo_inti[i] - 1;
  }

  int numj = 0;
  std::vector<std::string> rxt_names_read{};
  for (int m = 0; m < phtcnt; ++m) {
    if (lng_indexer_h(m) >= 0) {
      bool already_seen = false;
      for (int k = 0; k < m; ++k) {
        if (lng_indexer_h(k) == lng_indexer_h(m)) {
          already_seen = true;
          break;
        }
      }
      if (already_seen) continue;
      if (pht_alias_lst_2[m] != "NONE") {
        rxt_names_read.push_back(pht_alias_lst_2[m]);
      } else {
        rxt_names_read.push_back(rxt_names[m]);
      }
      numj++;
    }
  }

  auto table =
      scream::impl::read_photo_table(rsf_file, xs_long_file, rxt_names_read,
                                     numj, lng_indexer_h);
  modify_photo_table_pht_alias_mult_1(table);
  modify_photo_table_etfphot_e3smv3(table);
  return table;
}

int compute_nsamples(
    const Kokkos::View<const int*>& ntropopause,
    const int ncol,
    const int nlev,
    const bool above) {
  int nsamples = 0;
  Kokkos::parallel_reduce(
      "compute_nsamples", Kokkos::RangePolicy<TChem::exec_space>(0, ncol),
      KOKKOS_LAMBDA(const int icol, int& partial_sum) {
        partial_sum += above ? nlev - ntropopause(icol) : ntropopause(icol);
      },
      nsamples);
  return nsamples;
}

void compute_offsets(
    const Kokkos::View<const int*>& ntropopause,
    const int ncol,
    const int nlev,
    const Kokkos::View<int*>& offsets,
    const bool above) {
  Kokkos::parallel_scan(
      "compute_offsets", Kokkos::RangePolicy<TChem::exec_space>(0, ncol + 1),
      KOKKOS_LAMBDA(const int icol, int& partial_sum, const bool is_final) {
        if (is_final) offsets(icol) = partial_sum;
        if (icol < ncol)
          partial_sum += above ? nlev - ntropopause(icol) : ntropopause(icol);
      });
  Kokkos::fence();
}

void compute_sample_indices(
    const Kokkos::View<const int*>& ntropopause,
    const Kokkos::View<const int*>& offsets,
    const int ncol,
    const int nlev,
    const Kokkos::View<int*>& sample_icol,
    const Kokkos::View<int*>& sample_ilev,
    const bool above) {
  Kokkos::parallel_for(
      "fill_sample_indices", Kokkos::RangePolicy<TChem::exec_space>(0, ncol),
      KOKKOS_LAMBDA(const int icol) {
        const int lev_start = above ? ntropopause(icol) : 0;
        const int lev_end = above ? nlev : ntropopause(icol);
        const int offset = offsets(icol);
        for (int ilev = lev_start; ilev < lev_end; ++ilev) {
          const int isample = offset + (ilev - lev_start);
          sample_icol(isample) = icol;
          sample_ilev(isample) = ilev;
        }
      });
  Kokkos::fence();
}

}  // namespace tchem
}  // namespace scream
