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
