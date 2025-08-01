#include "catch2/catch.hpp"

#include "shoc_unit_tests_common.hpp"
#include "shoc_functions.hpp"
#include "shoc_test_data.hpp"
#include "physics/share/physics_constants.hpp"
#include "share/eamxx_types.hpp"
#include "share/util/eamxx_setup_random_test.hpp"

#include "ekat/ekat_pack.hpp"
#include "ekat/util/ekat_arch.hpp"
#include "ekat/kokkos/ekat_kokkos_utils.hpp"

#include <algorithm>
#include <array>
#include <random>
#include <thread>

namespace scream {
namespace shoc {
namespace unit_test {

template <typename D>
struct UnitWrap::UnitTest<D>::TestCompShocMixLength : public UnitWrap::UnitTest<D>::Base {

  void run_property()
  {
    static constexpr Int shcol    = 3;
    static constexpr Int nlev     = 5;

    // Tests for the SHOC function:
    //   compute_shoc_mix_shoc_length

    // Multi column test will verify that 1) mixing length increases
    // with height given brunt vaisalla frequency and
    // TKE are constant with height and 2) Columns with larger
    // TKE values produce a larger length scale value

    // Define TKE [m2/s2] that will be used for each column
    static constexpr Real tke_cons = 0.1;
    // Define the brunt vasailla frequency [s-1]
    static constexpr Real brunt_cons = 0.001;
    // Define the assymptoic length [m]
    static constexpr Real l_inf = 100;
    // Define the heights on the zt grid [m]
    static constexpr Real zt_grid[nlev] = {5000, 3000, 2000, 1000, 500};

    // Default SHOC formulation, not 1.5 TKE closure assumptions
    const bool shoc_1p5tke = false;

    // Initialize data structure for bridging to F90
    ComputeShocMixShocLengthData SDS(shcol, nlev, shoc_1p5tke);

    // Test that the inputs are reasonable.
    // For this test shcol MUST be at least 2
    REQUIRE( (SDS.shcol == shcol && SDS.nlev == nlev) );
    REQUIRE(shcol > 1);

    // Fill in test data on zt_grid.
    for(Int s = 0; s < shcol; ++s) {
      SDS.l_inf[s] = l_inf;
      for(Int n = 0; n < nlev; ++n) {
        const auto offset = n + s * nlev;

        // for the subsequent columns, increase
        //  the amount of TKE
        SDS.tke[offset] = (1.0+s)*tke_cons;
        SDS.brunt[offset] = brunt_cons;
        SDS.zt_grid[offset] = zt_grid[n];
        // do not consider below for default SHOC
	SDS.tk[offset] = 0;
	SDS.dz_zt[offset] = 0;
      }
    }

    // Check that the inputs make sense

    // Be sure that relevant variables are greater than zero
    for(Int s = 0; s < shcol; ++s) {
      REQUIRE(SDS.l_inf[s] > 0);
      for(Int n = 0; n < nlev; ++n) {
        const auto offset = n + s * nlev;
        REQUIRE(SDS.tke[offset] > 0);
        REQUIRE(SDS.zt_grid[offset] > 0);
        if (s < shcol-1){
          // Verify that tke is larger column by column
          const auto offsets = n + (s+1) * nlev;
          REQUIRE(SDS.tke[offset] < SDS.tke[offsets]);
        }
      }

      // Check that zt increases upward
      for(Int n = 0; n < nlev - 1; ++n) {
        const auto offset = n + s * nlev;
        REQUIRE(SDS.zt_grid[offset + 1] - SDS.zt_grid[offset] < 0);
      }
    }

    // Call the C++ implementation
    compute_shoc_mix_shoc_length(SDS);

    // Check the results
    for(Int s = 0; s < shcol; ++s) {
      for(Int n = 0; n < nlev; ++n) {
        const auto offset = n + s * nlev;
        // Validate shoc_mix greater than zero everywhere
        REQUIRE(SDS.shoc_mix[offset] > 0);
        if (s < shcol-1){
          // Verify that mixing length increases column by column
          const auto offsets = n + (s+1) * nlev;
          REQUIRE(SDS.shoc_mix[offset] < SDS.shoc_mix[offsets]);
        }
      }

      // Check that mixing length increases upward
      for(Int n = 0; n < nlev - 1; ++n) {
        const auto offset = n + s * nlev;
        REQUIRE(SDS.shoc_mix[offset + 1] - SDS.shoc_mix[offset] < 0);
      }
    }

    // 1.5 TKE test
    // Verify that length scale behaves as expected when 1.5 TKE closure
    //   assumptions are used. Will recycle all previous data, except we
    //   need to define dz, brunt vaisalla frequency, and tk.

    // Brunt Vaisalla frequency [s-1]
    static constexpr Real brunt_1p5[nlev] = {0.01,-0.01,0.01,-0.01,0.01};
    // Define the heights on the zt grid [m]
    static constexpr Real dz_zt_1p5[nlev] = {50, 100, 30, 20, 10};
    // Eddy viscocity [m2 s-1]
    static constexpr Real tk_cons_1p5 = 0.1;

    // Activate 1.5 TKE closure
    SDS.shoc_1p5tke = true;

    // Fill in test data on zt_grid.
    for(Int s = 0; s < shcol; ++s) {
      for(Int n = 0; n < nlev; ++n) {
        const auto offset = n + s * nlev;

        // do not consider below for default SHOC
	SDS.tk[offset] = tk_cons_1p5;
	SDS.dz_zt[offset] = dz_zt_1p5[n];
	SDS.brunt[offset] = brunt_1p5[n];
      }
    }

    // Call the C++ implementation
    compute_shoc_mix_shoc_length(SDS);

    // Check the result

    // Verify that if Brunt Vaisalla frequency is unstable that mixing length
    //  is equal to vertical grid spacing.  If brunt is stable, then verify that
    //  mixing length is less than the vertical grid spacing.
    for(Int s = 0; s < shcol; ++s) {
      for(Int n = 0; n < nlev; ++n) {
        const auto offset = n + s * nlev;
        if (SDS.brunt[offset] <= 0){
           REQUIRE(SDS.shoc_mix[offset] == SDS.dz_zt[offset]);
	}
	else{
	   REQUIRE(SDS.shoc_mix[offset] < SDS.dz_zt[offset]);
	   REQUIRE(SDS.shoc_mix[offset] >= 0.1*SDS.dz_zt[offset]);
	}

      }
    }

  }

  void run_bfb()
  {
    auto engine = Base::get_engine();

    ComputeShocMixShocLengthData SDS_baseline[] = {
      //               shcol, nlev
      ComputeShocMixShocLengthData(10, 71, false),
      ComputeShocMixShocLengthData(10, 12, false),
      ComputeShocMixShocLengthData(7,  16, false),
      ComputeShocMixShocLengthData(2, 7, false)
    };

    // Generate random input data
    for (auto& d : SDS_baseline) {
      d.randomize(engine);
    }

    // Create copies of data for use by cxx. Needs to happen before reads so that
    // inout data is in original state
    ComputeShocMixShocLengthData SDS_cxx[] = {
      ComputeShocMixShocLengthData(SDS_baseline[0]),
      ComputeShocMixShocLengthData(SDS_baseline[1]),
      ComputeShocMixShocLengthData(SDS_baseline[2]),
      ComputeShocMixShocLengthData(SDS_baseline[3]),
    };

    static constexpr Int num_runs = sizeof(SDS_baseline) / sizeof(ComputeShocMixShocLengthData);

    // Assume all data is in C layout

    // Read baseline data
    if (this->m_baseline_action == COMPARE) {
      for (auto& d : SDS_baseline) {
        d.read(Base::m_ifile);
      }
    }

    // Get data from cxx
    for (auto& d : SDS_cxx) {
      compute_shoc_mix_shoc_length(d);
    }

    // Verify BFB results, all data should be in C layout
    if (SCREAM_BFB_TESTING && this->m_baseline_action == COMPARE) {
      for (Int i = 0; i < num_runs; ++i) {
        ComputeShocMixShocLengthData& d_baseline = SDS_baseline[i];
        ComputeShocMixShocLengthData& d_cxx = SDS_cxx[i];
        for (Int k = 0; k < d_baseline.total(d_baseline.shoc_mix); ++k) {
          REQUIRE(d_baseline.shoc_mix[k] == d_cxx.shoc_mix[k]);
        }
      }
    } // SCREAM_BFB_TESTING
    else if (this->m_baseline_action == GENERATE) {
      for (Int i = 0; i < num_runs; ++i) {
        SDS_cxx[i].write(Base::m_ofile);
      }
    }
  }
};

}  // namespace unit_test
}  // namespace shoc
}  // namespace scream

namespace {

TEST_CASE("shoc_mix_length_property", "shoc")
{
  using TestStruct = scream::shoc::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestCompShocMixLength;

  TestStruct().run_property();
}

TEST_CASE("shoc_mix_length_bfb", "shoc")
{
  using TestStruct = scream::shoc::unit_test::UnitWrap::UnitTest<scream::DefaultDevice>::TestCompShocMixLength;

  TestStruct().run_bfb();
}

} // namespace
