#include <catch2/catch.hpp>

#include <Kokkos_Core.hpp>
#include <array>
#include <cmath>
#include <string>

#include "share/core/eamxx_types.hpp"

#include <mam4xx/mam4.hpp>
#include "../eamxx_tchem_atm_tchem_functions.hpp"
#include "share/scorpio_interface/eamxx_scorpio_interface.hpp"
#include <yaml-cpp/yaml.h>

namespace {

using Real = scream::Real;
using Device = scream::DefaultDevice;
using ExecSpace = Device::execution_space;
using HostSpace = Kokkos::HostSpace;
using KT = ekat::KokkosTypes<ExecSpace>;

using view_1d = typename KT::template view_1d<Real>;
using view_2d = typename KT::template view_2d<Real>;
using view_3d = typename KT::template view_3d<Real>;
using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;
using MemberType = TeamPolicy::member_type;

inline bool nearly_equal(const Real a, const Real b,
                         const Real rtol = 1e-8,
                         const Real atol = 1e-14) {
  return std::abs(a - b) <= atol + rtol * std::abs(b);
}

std::vector<Real> read_real_vector(const YAML::Node& node) {
  std::vector<Real> vals;
  vals.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    vals.push_back(node[i].as<Real>());
  }
  return vals;
}

std::vector<int> read_int_vector(const YAML::Node& node) {
  std::vector<int> vals;
  vals.reserve(node.size());
  for (std::size_t i = 0; i < node.size(); ++i) {
    vals.push_back(node[i].as<int>());
  }
  return vals;
}

} // anonymous namespace


TEST_CASE("tchem_photo_table_uci_raw_table_regression",
          "[mam4][photo][kokkos]") {
  using namespace scream;

  ekat::Comm comm(MPI_COMM_WORLD);
  scorpio::init_subsystem(comm);

  // Replace with your actual test-data paths.
  const std::string input_yaml_file = "jlong_input_ts_2016289.yaml";
  const std::string rsf_file = "/global/cfs/cdirs/e3sm/inputdata/atm/scream/mam4xx/photolysis/RSF_GT200nm_v3.0_c080811.nc";
  const std::string xs_long_file = "/global/cfs/cdirs/e3sm/inputdata/atm/scream/mam4xx/photolysis/temp_prs_GT200nm_JPL10_c130206.nc";

  REQUIRE(!rsf_file.empty());
  REQUIRE(!xs_long_file.empty());

  const YAML::Node root = YAML::LoadFile(input_yaml_file);
  REQUIRE(root["input"]);
  REQUIRE(root["input"]["fixed"]);
  const auto fixed = root["input"]["fixed"];

  // ---- Load the table under test -------------------------------------
  const auto photo_table = scream::tchem::read_photo_table_uci(rsf_file, xs_long_file);

  // ---- Pull reference data out of the YAML ----------------------------
  const auto nw_ref       = read_int_vector(fixed["nw"])[0];
  const auto numj_ref     = read_int_vector(fixed["numj"])[0];
  const auto shape_ref    = read_int_vector(fixed["shape_of_rsf_tab"]);
  REQUIRE(shape_ref.size() == 5);
  const int nw_shape       = shape_ref[0];
  const int nump_shape     = shape_ref[1];
  const int numsza_shape   = shape_ref[2];
  const int numcolo3_shape = shape_ref[3];
  const int numalb_shape   = shape_ref[4];

  const auto sza_ref       = read_real_vector(fixed["sza"]);
  const auto del_sza_ref   = read_real_vector(fixed["del_sza"]);
  const auto alb_ref       = read_real_vector(fixed["alb"]);
  const auto del_alb_ref   = read_real_vector(fixed["del_alb"]);
  const auto colo3_ref     = read_real_vector(fixed["colo3"]);
  const auto o3rat_ref     = read_real_vector(fixed["o3rat"]);
  const auto del_o3rat_ref = read_real_vector(fixed["del_o3rat"]);
  const auto etfphot_ref   = read_real_vector(fixed["etfphot"]);
  const auto prs_ref       = read_real_vector(fixed["prs"]);
  const auto dprs_ref      = read_real_vector(fixed["dprs"]);
  const auto rsf_tab_2d    = read_real_vector(fixed["rsf_tab_2d"]);
  const auto xsqy_h_2d    = read_real_vector(fixed["xsqy_2d"]);
  

  // ---- Sanity-check dimensions against the YAML's metadata -----------
  REQUIRE(photo_table.nw       == nw_ref);
  REQUIRE(photo_table.numj     == numj_ref);
  REQUIRE(photo_table.nw       == nw_shape);
  REQUIRE(photo_table.nump     == nump_shape);
  REQUIRE(photo_table.numsza   == numsza_shape);
  REQUIRE(photo_table.numcolo3 == numcolo3_shape);
  REQUIRE(photo_table.numalb   == numalb_shape);

  REQUIRE(static_cast<int>(sza_ref.size())       == photo_table.numsza);
  REQUIRE(static_cast<int>(del_sza_ref.size())   == photo_table.numsza - 1);
  REQUIRE(static_cast<int>(alb_ref.size())       == photo_table.numalb);
  REQUIRE(static_cast<int>(del_alb_ref.size())   == photo_table.numalb - 1);
  REQUIRE(static_cast<int>(colo3_ref.size())     == photo_table.nump);
  REQUIRE(static_cast<int>(o3rat_ref.size())     == photo_table.numcolo3);
  REQUIRE(static_cast<int>(del_o3rat_ref.size()) == photo_table.numcolo3 - 1);
  REQUIRE(static_cast<int>(etfphot_ref.size())   == photo_table.nw);
  REQUIRE(static_cast<int>(prs_ref.size())       == photo_table.np_xs);
  REQUIRE(static_cast<int>(dprs_ref.size())      == photo_table.np_xs - 1);
  REQUIRE(rsf_tab_2d.size() ==
          static_cast<std::size_t>(photo_table.nw) *
          static_cast<std::size_t>(photo_table.nump));

  // ---- Copy the 1D table views to host for comparison -----------------
  auto sza_h       = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.sza);
  auto del_sza_h    = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.del_sza);
  auto alb_h        = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.alb);
  auto del_alb_h    = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.del_alb);
  auto colo3_h      = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.colo3);
  auto o3rat_h      = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.o3rat);
  auto del_o3rat_h  = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.del_o3rat);
  auto etfphot_h    = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.etfphot);
  auto prs_h        = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.prs);
  auto dprs_h       = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.dprs);
  auto rsf_tab_h    = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.rsf_tab);
  auto xsqy_h    = Kokkos::create_mirror_view_and_copy(HostSpace(), photo_table.xsqy);

  SECTION("sza_matches_reference") {
    for (int i = 0; i < photo_table.numsza; ++i) {
      INFO("sza mismatch at i=" << i << ", computed=" << sza_h(i)
           << ", expected=" << sza_ref[i]);
      REQUIRE(nearly_equal(sza_h(i), sza_ref[i]));
    }
  }

  SECTION("del_sza_matches_reference") {
    for (int i = 0; i < photo_table.numsza - 1; ++i) {
      INFO("del_sza mismatch at i=" << i << ", computed=" << del_sza_h(i)
           << ", expected=" << del_sza_ref[i]);
      REQUIRE(nearly_equal(del_sza_h(i), del_sza_ref[i]));
    }
  }

  SECTION("alb_matches_reference") {
    for (int i = 0; i < photo_table.numalb; ++i) {
      INFO("alb mismatch at i=" << i << ", computed=" << alb_h(i)
           << ", expected=" << alb_ref[i]);
      REQUIRE(nearly_equal(alb_h(i), alb_ref[i]));
    }
  }

  SECTION("del_alb_matches_reference") {
    for (int i = 0; i < photo_table.numalb - 1; ++i) {
      INFO("del_alb mismatch at i=" << i << ", computed=" << del_alb_h(i)
           << ", expected=" << del_alb_ref[i]);
      REQUIRE(nearly_equal(del_alb_h(i), del_alb_ref[i]));
    }
  }

  SECTION("colo3_matches_reference") {
    for (int i = 0; i < photo_table.nump; ++i) {
      INFO("colo3 mismatch at i=" << i << ", computed=" << colo3_h(i)
           << ", expected=" << colo3_ref[i]);
      REQUIRE(nearly_equal(colo3_h(i), colo3_ref[i]));
    }
  }

  SECTION("o3rat_matches_reference") {
    for (int i = 0; i < photo_table.numcolo3; ++i) {
      INFO("o3rat mismatch at i=" << i << ", computed=" << o3rat_h(i)
           << ", expected=" << o3rat_ref[i]);
      // printf("o3rat mismatch at i=%d, computed=%g, expected=%g\n",
      //  i, o3rat_h(i), o3rat_ref[i]);     
      REQUIRE(nearly_equal(o3rat_h(i), o3rat_ref[i]));
    }
  }

  SECTION("del_o3rat_matches_reference") {
    for (int i = 0; i < photo_table.numcolo3 - 1; ++i) {
      INFO("del_o3rat mismatch at i=" << i << ", computed=" << del_o3rat_h(i)
           << ", expected=" << del_o3rat_ref[i]);
      // printf("del_o3rat mismatch at i=%d, computed=%g, expected=%g\n",
      //  i, del_o3rat_h(i), del_o3rat_ref[i]);     
      REQUIRE(nearly_equal(del_o3rat_h(i), del_o3rat_ref[i]));
    }
  }
  
  SECTION("prs_matches_reference") {
    for (int i = 0; i < photo_table.np_xs; ++i) {
      INFO("prs mismatch at i=" << i << ", computed=" << prs_h(i)
           << ", expected=" << prs_ref[i]);
      REQUIRE(nearly_equal(prs_h(i), prs_ref[i]));
    }
  }

  SECTION("dprs_matches_reference") {
    for (int i = 0; i < photo_table.np_xs - 1; ++i) {
      INFO("dprs mismatch at i=" << i << ", computed=" << dprs_h(i)
           << ", expected=" << dprs_ref[i]);
      REQUIRE(nearly_equal(dprs_h(i), dprs_ref[i]));
    }
  }

  SECTION("etfphot_matches_reference") {
    for (int i = 0; i < photo_table.nw; ++i) {
      INFO("etfphot mismatch at i=" << i << ", computed=" << etfphot_h(i)
           << ", expected=" << etfphot_ref[i]);
      // printf("etfphot mismatch at i=%d, computed=%g, expected=%g\n",
      //  i, etfphot_h(i), etfphot_ref[i]);     
      REQUIRE(nearly_equal(etfphot_h(i), etfphot_ref[i]));
    }
  }

  // ---- rsf_tab(:,:,0,0,0) slice, unflattened from Fortran column-major
  // order (k = i + j*nw, i.e. the wavelength index i varies fastest).
  SECTION("xsqy_h_corner_slice_matches_reference") {
    const int nw   = photo_table.nw;
    const int nump = photo_table.nump;
    int count = 0;
    for (int d2 = 0; d2 < nump; ++d2) {
      for (int d1 = 0; d1 < nw; ++d1) {
        const auto computed = rsf_tab_h(d1, d2, 0, 0, 0);;
        const auto expected = rsf_tab_2d[count];
        count++;
        INFO("rsf_tab mismatch at (i=" << d1 << ", j=" << d2
               << "), computed=" << computed << ", expected=" << expected);
        REQUIRE(nearly_equal(computed, expected,1e-6));
        // std::cout << "k=" << d1
        // std::cout << "computed=" << computed
        //       << ", expected=" << expected
        //       << ", diff=" << (computed - expected)
        //       << "\n";  
  }
}
  }

    SECTION("xsqy_h_corner_slice_matches_reference") {
    const int nw   = photo_table.nw;
    const int numj = photo_table.numj;
    int count = 0;
    std::cout << " xsqy_h_corner_slice_matches_reference \n";
    for (int d2 = 0; d2 < nw; ++d2) {
      // std::cout << "d2 = "<< d2<<"\n";
      for (int d1 = 0; d1 < numj; ++d1) {
        const auto computed = xsqy_h(d1, d2, 0, 0);;
        const auto expected = xsqy_h_2d[count];
        count++;
        // std::cout << "computed=" << computed
        //       << ", expected=" << expected
        //       << ", diff=" << (computed - expected)
        //       << "\n"; 
        INFO("rsf_tab mismatch at (i=" << d1 << ", j=" << d2
               << "), computed=" << computed << ", expected=" << expected);
        REQUIRE(nearly_equal(computed, expected,0));
        // std::cout << "k=" << d1
        // std::cout << "computed=" << computed
        //       << ", expected=" << expected
        //       << ", diff=" << (computed - expected)
        //       << "\n";  
  }
}
  }

  scorpio::finalize_subsystem();
}


TEST_CASE("tchem_photo_table_kernel_single_column_nlev72_regression",
          "[mam4][photo][kokkos]") {
  constexpr int ncol = 1;
  constexpr int nlev = mam4::nlev;
  constexpr int nref = 22;
  using namespace scream;

  ekat::Comm comm(MPI_COMM_WORLD);
  scorpio::init_subsystem(comm);

  // Replace with your actual test-data paths.
  const std::string input_yaml_file = "table_photo_input_ts_2016289.yaml";
  const std::string rsf_file = "/global/cfs/cdirs/e3sm/inputdata/atm/scream/mam4xx/photolysis/RSF_GT200nm_v3.0_c080811.nc";
  const std::string xs_long_file = "/global/cfs/cdirs/e3sm/inputdata/atm/scream/mam4xx/photolysis/temp_prs_GT200nm_JPL10_c130206.nc";
  
  REQUIRE(!rsf_file.empty());
  REQUIRE(!xs_long_file.empty());

  const YAML::Node root = YAML::LoadFile(input_yaml_file);
  REQUIRE(root["input"]);
  REQUIRE(root["input"]["fixed"]);

  const auto fixed = root["input"]["fixed"];
  // Load photo table.
  const auto photo_table = scream::tchem::read_photo_table_uci(rsf_file, xs_long_file);
  const int work_len = mam4::mo_photo::get_photo_table_work_len(photo_table);
  const int npht = mam4::mo_photo::phtcnt;

  REQUIRE(work_len > 0);
  REQUIRE(npht >= nref);

  // Allocate views.
  view_2d work_photo_table("work_photo_table", ncol, work_len);
  view_2d pmid("pmid", ncol, nlev);
  view_2d pdel("pdel", ncol, nlev);
  view_2d temper("temper", ncol, nlev);
  view_2d o3col("o3col", ncol, nlev);
  view_1d zen_angle("zen_angle", ncol);
  view_1d srf_alb("srf_alb", ncol);
  view_2d qc("qc", ncol, nlev);
  view_2d cld("cld", ncol, nlev);
  view_3d photo("photo", ncol, nlev, npht);

  Kokkos::deep_copy(work_photo_table, 0.0);
  Kokkos::deep_copy(photo, 0.0);

  // Values from the Python file, repeated for all 72 levels.
  const auto pmid_vals   = read_real_vector(fixed["pmid"]);
  const auto pdel_vals   = read_real_vector(fixed["pdel"]);
  const auto temper_vals = read_real_vector(fixed["temper"]);
  //CHECK
  const auto o3col_vals  = read_real_vector(fixed["col_dens_1"]);
  const auto lwc_vals    = read_real_vector(fixed["lwc"]);
  const auto cloud_vals  = read_real_vector(fixed["clouds"]);
  const auto zen_vals    = read_real_vector(fixed["zen_angle"]);
  const auto alb_vals    = read_real_vector(fixed["srf_alb"]);
  const auto esfact_vals = read_real_vector(fixed["esfact"]);
  //reference: I save this reference in the input section
  const auto photo_ref = read_real_vector(fixed["photos"]);
  
  REQUIRE(pmid_vals.size()   >= nlev);
  REQUIRE(pdel_vals.size()   >= nlev);
  REQUIRE(temper_vals.size() >= nlev);
  REQUIRE(o3col_vals.size()  >= nlev);
  REQUIRE(lwc_vals.size()    >= nlev);
  REQUIRE(cloud_vals.size()  >= nlev);
  REQUIRE(zen_vals.size()    >= 1);
  REQUIRE(alb_vals.size()    >= 1);
  REQUIRE(esfact_vals.size() >= 1);

  const Real zen_val = zen_vals[0];
  const Real alb_val = alb_vals[0];
  const Real esfact  = esfact_vals[0];

  auto pmid_h   = Kokkos::create_mirror_view(pmid);
  auto pdel_h   = Kokkos::create_mirror_view(pdel);
  auto temper_h = Kokkos::create_mirror_view(temper);
  auto o3col_h  = Kokkos::create_mirror_view(o3col);
  auto zen_h    = Kokkos::create_mirror_view(zen_angle);
  auto alb_h    = Kokkos::create_mirror_view(srf_alb);
  auto qc_h     = Kokkos::create_mirror_view(qc);
  auto cld_h    = Kokkos::create_mirror_view(cld);

  for (int k = 0; k < nlev; ++k) {
    // printf("pmid_vals %e \n", pmid_vals[k]);
    pmid_h(0, k)   = pmid_vals[k];
    pdel_h(0, k)   = pdel_vals[k];
    temper_h(0, k) = temper_vals[k];
    o3col_h(0, k)  = o3col_vals[k];
    qc_h(0, k)     = lwc_vals[k];
    cld_h(0, k)    = cloud_vals[k];
    // printf("cloud_vals %e \n", cloud_vals[k]);
  }
  zen_h(0) = zen_val;
  alb_h(0) = alb_val;

  Kokkos::deep_copy(pmid, pmid_h);
  Kokkos::deep_copy(pdel, pdel_h);
  Kokkos::deep_copy(temper, temper_h);
  Kokkos::deep_copy(o3col, o3col_h);
  Kokkos::deep_copy(zen_angle, zen_h);
  Kokkos::deep_copy(srf_alb, alb_h);
  Kokkos::deep_copy(qc, qc_h);
  Kokkos::deep_copy(cld, cld_h);

  // Launch one-column kernel.
  TeamPolicy policy(ncol, Kokkos::AUTO());

  Kokkos::parallel_for(
    "unit_test_table_photo_nlev72", policy,
    KOKKOS_LAMBDA(const MemberType& team) {
      const int icol = team.league_rank();

      const auto work_icol = ekat::subview(work_photo_table, icol);

      mam4::mo_photo::PhotoTableWorkArrays photo_work_arrays;
      mam4::mo_photo::set_photo_table_work_arrays(photo_table, work_icol,
                                                  photo_work_arrays);
      team.team_barrier();

      const auto pmid_col  = ekat::subview(pmid, icol);
      const auto pdel_col  = ekat::subview(pdel, icol);
      const auto t_col     = ekat::subview(temper, icol);
      const auto o3_col    = ekat::subview(o3col, icol);
      const auto qc_col    = ekat::subview(qc, icol);
      const auto cld_col   = ekat::subview(cld, icol);
      const auto photo_col = ekat::subview(photo, icol);

      mam4::mo_photo::table_photo(team, photo_col, pmid_col, pdel_col, t_col,
                                  o3_col, zen_angle(icol), srf_alb(icol),
                                  qc_col, cld_col, esfact, photo_table,
                                  photo_work_arrays);
    });
  Kokkos::fence();

  // const std::array<Real, nref> expected = {{
  //   0.10384950618677957E-003,
  //   0.00000000000000000E+000,
  //   0.12486522593829268E-004,
  //   0.64723189182607361E-004,
  //   0.88690605167060716E-004,
  //   0.10211860107866727E-004,
  //   0.10211860107866727E-004,
  //   0.13018752803418717E-001,
  //   0.20679806810851048E+000,
  //   0.10945949633620182E-001,
  //   0.61631151464815008E-004,
  //   0.17414432160963850E-007,
  //   0.14203223665893713E-005,
  //   0.28412855422172683E-005,
  //   0.21666703381029496E-004,
  //   0.28473882789617557E-004,
  //   0.12167127640722079E-005,
  //   0.21488166054649110E-005,
  //   0.16537890208910176E-004,
  //   0.52075011213674859E-005,
  //   0.52075011213674859E-005,
  //   0.52075011213674859E-005
  // }};

  auto photo_h = Kokkos::create_mirror_view_and_copy(HostSpace(), photo);

  SECTION("all_outputs_are_finite") {
    for (int k = 0; k < nlev; ++k) {
      for (int j = 0; j < nref; ++j) {
        INFO("Non-finite output at k=" << k << ", j=" << j
             << ", value=" << photo_h(0, k, j));
        REQUIRE(std::isfinite(photo_h(0, k, j)));
      }
    }
  }

  // SECTION("all_72_levels_match_each_other_for_identical_inputs") {
  //   for (int k = 1; k < nlev; ++k) {
  //     for (int j = 0; j < nref; ++j) {
  //       INFO("Mismatch across repeated levels at k=" << k << ", j=" << j
  //            << ", photo(0," << k << "," << j << ")=" << photo_h(0, k, j)
  //            << ", photo(0,0," << j << ")=" << photo_h(0, 0, j));
  //       REQUIRE(nearly_equal(photo_h(0, k, j), photo_h(0, 0, j), 1e-8, 1e-14));
  //     }
  //   }
  // }

  // SECTION("level_0_matches_python_reference") {
  //   for (int j = 0; j < nref; ++j) {
  //     std::cout << "j=" << j
  //             << ", computed=" << photo_h(0, 0, j)
  //             << ", expected=" << expected[j]
  //             << ", diff=" << (photo_h(0, 0, j) - expected[j])
  //             << "\n";

  //     INFO("Reference mismatch at j=" << j
  //          << ", computed=" << photo_h(0, 0, j)
  //          << ", expected=" << expected[j]);
  //     // REQUIRE(nearly_equal(photo_h(0, 0, j), expected[j], 1e-8, 1e-14));
  //   }
  // }
  SECTION("compare_against_reference_when_available") {
  REQUIRE(photo_ref.size() == static_cast<std::size_t>(nlev * nref));

  int count = 0;
  for (int d2 = 0; d2 < nref; ++d2) {
    // std::cout << "j=" << d2<< "\n"; 
    for (int d1 = 0; d1 < nlev; ++d1) {
      const auto computed = photo_h(0, d1, d2);
      const auto expected = photo_ref[count];
      count++;
      if (d1==0)
      {  // std::cout << "k=" << d1
        Real diff=computed - expected;       
        Real rel = diff/expected;
        std::cout <<"ireac "<< d2 << " computed=" << computed
              << ", expected=" << expected
              << ", diff=" << (diff)
              << ", ref=" << (rel)
              << "\n";  
      }

    }
  }


    // for (int k = 0; k < nlev; ++k) {
    //   std::cout << "k=" << k<< "\n"; 
    //   for (int j = 0; j < nref; ++j) {
    //     const int idx = j + k * nlev;
    //     // const int idx = k * nref + j; // assumes level-major flattening
    //     const auto computed = photo_h(0, k, j);
    //     const auto expected = photo_ref[idx];

    //     INFO("Mismatch at k=" << k << ", j=" << j
    //          << ", computed=" << computed
    //          << ", expected=" << expected);
    //     std::cout << "j=" << j
    //           << ", computed=" << computed
    //           << ", expected=" << expected
    //           << ", diff=" << (computed - expected)
    //           << "\n";     
    //     // REQUIRE(nearly_equal(computed, expected, 1e-8, 1e-14));
    //   }
    // }
  }

  // SECTION("all_72_levels_match_python_reference") {
  //   for (int k = 0; k < nlev; ++k) {
  //     for (int j = 0; j < nref; ++j) {
  //       INFO("Reference mismatch at k=" << k << ", j=" << j
  //            << ", computed=" << photo_h(0, k, j)
  //            << ", expected=" << expected[j]);
  //       REQUIRE(nearly_equal(photo_h(0, k, j), expected[j], 1e-8, 1e-14));
  //     }
  //   }
  // }

  scorpio::finalize_subsystem();
}
