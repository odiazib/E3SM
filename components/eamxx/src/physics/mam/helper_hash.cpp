#include "share/util/eamxx_bfbhash.hpp"

namespace scream::impl {
  template<typename ViewType>
  void compute_and_print_hash(ViewType& view,
                              const std::string& nickname,
                              const int iforcing,
                              const int isec ) {
    bfbhash::HashType accum = 0, gaccum = 0;

    // Perform parallel reduction using Kokkos
    Kokkos::parallel_reduce(
        "bfb_reduce", view.extent(0),
        KOKKOS_LAMBDA(const int icol, bfbhash::HashType& accum) {
            for (int i = 0; i < view.extent(1); ++i) {
                bfbhash::hash(view(icol, i), accum);
            }
        },
        bfbhash::HashReducer<>(accum)
    );

    Kokkos::fence();

    // Perform MPI all-reduce operation
    bfbhash::all_reduce_HashType(MPI_COMM_WORLD, &accum, &gaccum, 1);

    // Create communicator and print the result if the current process is the root
    ekat::Comm comm(MPI_COMM_WORLD);
    if (comm.am_i_root()) {
        std::cout << "bfbhash> "<<nickname<< iforcing << "_sector_" << isec << " "
                  << std::hex << std::setfill('0') << std::setw(16) << gaccum
                  << std::dec << std::endl;
    }
  };
  template<typename ViewType>
  void compute_and_print_diff(ViewType& view1,
                              ViewType& view2,
                              const std::string& nickname,
                                    const int iforcing,
                                    const int isec ) {
    Real diff = 0;

    // Perform parallel reduction using Kokkos
    Kokkos::parallel_reduce(
        "bfb_reduce", view1.extent(0),
        KOKKOS_LAMBDA(const int icol, Real& diff) {
            for (int i = 0; i < view1.extent(1); ++i) {
                diff += Kokkos::fabs(view1(icol, i)-view2(icol, i));
            }
        },
        diff
    );

    Kokkos::fence();
    ekat::Comm comm(MPI_COMM_WORLD);
    if (comm.am_i_root()) {
        std::cout << "diff> "<<nickname<< iforcing << "_sector_" << isec << " "
                   <<std::setprecision(16) << diff << std::endl;
    }
  };
  template<typename ViewType>
  void compute_and_print_sum(ViewType& view,
                             const std::string& nickname,
                                    const int iforcing,
                                    const int isec ) {
    Real suma = 0;

    // Perform parallel reduction using Kokkos
    Kokkos::parallel_reduce(
        "bfb_reduce", view.extent(0),
        KOKKOS_LAMBDA(const int icol, Real& suma) {
            for (int i = 0; i < view.extent(1); ++i) {
                suma += view(icol, i);
            }
        },
        suma
    );

    Kokkos::fence();
    ekat::Comm comm(MPI_COMM_WORLD);
    if (comm.am_i_root()) {
        std::cout << "sum> "<<nickname<< iforcing << "_sector_" << isec << " "
                   <<std::setprecision(16) << suma << std::endl;
    }
  };
#if 0
  for(int i = 0; i < extcnt; ++i) {
    const int nsectors       = forcings_[i].nsectors;
    const int frc_ndx        = forcings_[i].frc_ndx;
    const auto file_alt_data = forcings_[i].file_alt_data;
    for(int isec = 0; isec < forcings_[i].nsectors; ++isec) {
     const auto& field = forcings_[i].fields[isec];
     const auto& field2 = test_forcings_[i].fields[isec];
     scream::impl::compute_and_print_hash(field, "forc_old_",i,isec);
     scream::impl::compute_and_print_hash(field2,"forc_new_",i,isec);
     scream::impl::compute_and_print_diff(field,field2,"DIFF_",i,isec);
     scream::impl::compute_and_print_sum(field, "forc_old_",i,isec);
     scream::impl::compute_and_print_sum(field2,"forc_new_",i,isec);
  }  // extcnt for loop
  }
#endif
}
