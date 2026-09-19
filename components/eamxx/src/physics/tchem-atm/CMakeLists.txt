if (SCREAM_CIME_BUILD)
  # PROJECT_SOURCE_DIR is SCREAM_ROOT/components
  set(EXTERNALS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../externals")
else()
  # PROJECT_SOURCE_DIR is SCREAM_ROOT/components/eamxx
  set(EXTERNALS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../externals")
endif()

# yaml-cpp dependency for TChem-atm (bundled, must be added before Tines)
if (NOT TARGET yaml-cpp)
  set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests within E3SM" FORCE)
  set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools within E3SM" FORCE)
  set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib within E3SM" FORCE)
  set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install exports within E3SM" FORCE)
  add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/external/Tines/ext/yaml
                   ${CMAKE_BINARY_DIR}/externals/tchem-atm/external/yaml-cpp)
endif()
# Tines checks for a target named "yaml"; create alias if needed
if (TARGET yaml-cpp AND NOT TARGET yaml)
  add_library(yaml ALIAS yaml-cpp)
endif()

# Tines dependency for TChem-atm
if (NOT TARGET Tines::tines)
  set(TINES_ENABLE_TEST OFF CACHE BOOL "Disable Tines tests within E3SM" FORCE)
  set(TINES_ENABLE_EXAMPLE OFF CACHE BOOL "Disable Tines examples within E3SM" FORCE)
  set(TINES_ENABLE_INSTALL OFF CACHE BOOL "Disable Tines install exports within E3SM" FORCE)
  add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/external/Tines/src ${CMAKE_BINARY_DIR}/externals/tchem-atm/external/Tines)
  if (TARGET tines AND NOT TARGET Tines::tines)
    add_library(Tines::tines ALIAS tines)
  endif()
endif()

# Sundials dependency for TChem-atm
# Option to enable CVODE batch solver support (requires Sundials with Kokkos)
option(EAMXX_TCHEM_ENABLE_SUNDIALS "Enable Sundials CVODE batch solver for TChem-atm" ON)

if (EAMXX_TCHEM_ENABLE_SUNDIALS)
  # ===========================================================================
  # Step 1: Build KokkosKernels if not already available
  # (Required by Sundials Kokkos dense matrix/linear solver)
  # ===========================================================================
  if (NOT TARGET Kokkos::kokkoskernels AND NOT TARGET KokkosKernels::kokkoskernels)
    message(STATUS "Building KokkosKernels from source for TChem-atm CVODE support")
    
    # KokkosKernels build options - minimal build
    set(KokkosKernels_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_BENCHMARK OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_DOCS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_DOUBLE ON CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_FLOAT OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_COMPLEX_DOUBLE OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_INST_COMPLEX_FLOAT OFF CACHE BOOL "" FORCE)
    # Enable batched components needed for Sundials
    set(KokkosKernels_ENABLE_TPL_BLAS OFF CACHE BOOL "" FORCE)
    set(KokkosKernels_ENABLE_TPL_LAPACK OFF CACHE BOOL "" FORCE)
    
    add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/external/kokkos-kernels 
                     ${CMAKE_BINARY_DIR}/externals/tchem-atm/external/kokkos-kernels)
    
    # Create namespaced alias if needed
    if (TARGET kokkoskernels AND NOT TARGET Kokkos::kokkoskernels)
      add_library(Kokkos::kokkoskernels ALIAS kokkoskernels)
    endif()
    if (TARGET kokkoskernels AND NOT TARGET KokkosKernels::kokkoskernels)
      add_library(KokkosKernels::kokkoskernels ALIAS kokkoskernels)
    endif()
  endif()

  # ===========================================================================
  # Step 2: Build Sundials with Kokkos and KokkosKernels support
  # ===========================================================================
  if (NOT TARGET SUNDIALS::cvode AND NOT TARGET sundials_cvode)
    message(STATUS "Building Sundials from source for TChem-atm CVODE support")
    
    # Sundials build options - only build what we need
    set(BUILD_CVODE ON CACHE BOOL "Build CVODE" FORCE)
    set(BUILD_ARKODE OFF CACHE BOOL "Build ARKODE" FORCE)
    set(BUILD_CVODES OFF CACHE BOOL "Build CVODES" FORCE)
    set(BUILD_IDA OFF CACHE BOOL "Build IDA" FORCE)
    set(BUILD_IDAS OFF CACHE BOOL "Build IDAS" FORCE)
    set(BUILD_KINSOL OFF CACHE BOOL "Build KINSOL" FORCE)
    
    # Disable Sundials tests/examples/benchmarks
    set(SUNDIALS_BUILD_WITH_MONITORING OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable Sundials tests" FORCE)
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "Disable Sundials C examples" FORCE)
    set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "Disable Sundials C++ examples" FORCE)
    set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable Sundials benchmarks" FORCE)
    
    # Enable Kokkos and KokkosKernels support in Sundials
    set(ENABLE_KOKKOS ON CACHE BOOL "Enable Kokkos in Sundials" FORCE)
    set(ENABLE_KOKKOS_KERNELS ON CACHE BOOL "Enable KokkosKernels in Sundials" FORCE)
    
    # Tell Sundials that Kokkos and KokkosKernels work
    set(KOKKOS_WORKS TRUE CACHE BOOL "Kokkos works with Sundials" FORCE)
    set(KOKKOSKERNELS_WORKS TRUE CACHE BOOL "KokkosKernels works with Sundials" FORCE)
    set(Kokkos_FOUND TRUE CACHE BOOL "Kokkos is found" FORCE)
    set(KokkosKernels_FOUND TRUE CACHE BOOL "KokkosKernels is found" FORCE)
    
    # Build Kokkos-based modules
    set(BUILD_NVECTOR_KOKKOS ON CACHE BOOL "Build Kokkos NVector" FORCE)
    set(BUILD_SUNMATRIX_KOKKOSDENSE ON CACHE BOOL "Build Kokkos Dense Matrix" FORCE)
    set(BUILD_SUNLINSOL_KOKKOSDENSE ON CACHE BOOL "Build Kokkos Dense Linear Solver" FORCE)
    
    # Disable other TPLs we don't need
    set(ENABLE_MPI OFF CACHE BOOL "" FORCE)
    set(ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(ENABLE_PTHREAD OFF CACHE BOOL "" FORCE)
    set(ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(ENABLE_HIP OFF CACHE BOOL "" FORCE)
    set(ENABLE_SYCL OFF CACHE BOOL "" FORCE)
    set(ENABLE_LAPACK OFF CACHE BOOL "" FORCE)
    set(ENABLE_KLU OFF CACHE BOOL "" FORCE)
    set(ENABLE_SUPERLUMT OFF CACHE BOOL "" FORCE)
    set(ENABLE_SUPERLUDIST OFF CACHE BOOL "" FORCE)
    set(ENABLE_PETSC OFF CACHE BOOL "" FORCE)
    set(ENABLE_HYPRE OFF CACHE BOOL "" FORCE)
    set(ENABLE_TRILINOS OFF CACHE BOOL "" FORCE)
    set(ENABLE_RAJA OFF CACHE BOOL "" FORCE)
    set(ENABLE_GINKGO OFF CACHE BOOL "" FORCE)
    set(ENABLE_MAGMA OFF CACHE BOOL "" FORCE)
    set(ENABLE_ONEMKL OFF CACHE BOOL "" FORCE)
    set(ENABLE_CALIPER OFF CACHE BOOL "" FORCE)
    set(ENABLE_ADIAK OFF CACHE BOOL "" FORCE)
    
    # Sundials source directory
    set(SUNDIALS_SOURCE_DIR "${EXTERNALS_SOURCE_DIR}/tchem-atm/external/Sundials")
    
    # Create custom module overrides for Kokkos and KokkosKernels
    set(SUNDIALS_CUSTOM_MODULE_PATH "${CMAKE_BINARY_DIR}/externals/sundials-cmake-override")
    file(MAKE_DIRECTORY ${SUNDIALS_CUSTOM_MODULE_PATH})
    
    # Override SundialsKokkos.cmake
    file(WRITE "${SUNDIALS_CUSTOM_MODULE_PATH}/SundialsKokkos.cmake" "
# Override SundialsKokkos.cmake - use existing Kokkos::kokkos target
if(NOT DEFINED SUNDIALS_KOKKOS_INCLUDED)
  set(SUNDIALS_KOKKOS_INCLUDED)
else()
  return()
endif()

if(NOT TARGET Kokkos::kokkos)
  message(FATAL_ERROR \"Kokkos::kokkos target not found. EAMxx should provide this.\")
endif()

message(STATUS \"Using existing Kokkos::kokkos target for Sundials\")
set(KOKKOS_EXAMPLES_BACKENDS \"SERIAL\" CACHE STRING \"Kokkos backends\" FORCE)
set(KOKKOS_WORKS TRUE CACHE BOOL \"Kokkos works with SUNDIALS\" FORCE)
set(Kokkos_FOUND TRUE)
")
    
    # Override SundialsKokkosKernels.cmake
    file(WRITE "${SUNDIALS_CUSTOM_MODULE_PATH}/SundialsKokkosKernels.cmake" "
# Override SundialsKokkosKernels.cmake - use existing KokkosKernels target
if(NOT DEFINED SUNDIALS_KOKKOSKERNELS_INCLUDED)
  set(SUNDIALS_KOKKOSKERNELS_INCLUDED)
else()
  return()
endif()

# Check for either target name
if(TARGET Kokkos::kokkoskernels)
  message(STATUS \"Using existing Kokkos::kokkoskernels target for Sundials\")
elseif(TARGET KokkosKernels::kokkoskernels)
  message(STATUS \"Using existing KokkosKernels::kokkoskernels target for Sundials\")
  # Create alias for Sundials
  if(NOT TARGET Kokkos::kokkoskernels)
    add_library(Kokkos::kokkoskernels ALIAS KokkosKernels::kokkoskernels)
  endif()
elseif(TARGET kokkoskernels)
  message(STATUS \"Using existing kokkoskernels target for Sundials\")
  if(NOT TARGET Kokkos::kokkoskernels)
    add_library(Kokkos::kokkoskernels ALIAS kokkoskernels)
  endif()
else()
  message(FATAL_ERROR \"KokkosKernels target not found.\")
endif()

set(KOKKOSKERNELS_WORKS TRUE CACHE BOOL \"KokkosKernels works with SUNDIALS\" FORCE)
set(KokkosKernels_FOUND TRUE)
")
    
    # Add Sundials with our module path taking precedence
    list(INSERT CMAKE_MODULE_PATH 0 "${SUNDIALS_CUSTOM_MODULE_PATH}")
    
    add_subdirectory(${SUNDIALS_SOURCE_DIR} ${CMAKE_BINARY_DIR}/externals/sundials)
    
    # Remove our override path after Sundials is configured
    list(REMOVE_AT CMAKE_MODULE_PATH 0)
    
    # Create SUNDIALS:: aliases if needed
    if (TARGET sundials_cvode AND NOT TARGET SUNDIALS::cvode)
      add_library(SUNDIALS::cvode ALIAS sundials_cvode)
    endif()
    if (TARGET sundials_nveckokkos AND NOT TARGET SUNDIALS::nveckokkos)
      add_library(SUNDIALS::nveckokkos ALIAS sundials_nveckokkos)
    endif()
    if (TARGET sundials_sunmatrixkokkosdense AND NOT TARGET SUNDIALS::sunmatrixkokkosdense)
      add_library(SUNDIALS::sunmatrixkokkosdense ALIAS sundials_sunmatrixkokkosdense)
    endif()
    if (TARGET sundials_sunlinsolkokkosdense AND NOT TARGET SUNDIALS::sunlinsolkokkosdense)
      add_library(SUNDIALS::sunlinsolkokkosdense ALIAS sundials_sunlinsolkokkosdense)
    endif()
  endif()
  
  set(TCHEM_ATM_ENABLE_SUNDIALS ON CACHE BOOL "Enable TChem-atm Sundials support" FORCE)
  # Also tell TChem-atm that KokkosKernels is available
  set(TCHEM_ATM_ENABLE_KOKKOSKERNELS ON CACHE BOOL "Enable TChem-atm KokkosKernels support" FORCE)
else()
  set(TCHEM_ATM_ENABLE_SUNDIALS OFF CACHE BOOL "Disable TChem-atm Sundials support" FORCE)
endif()

# TChem-atm external package
set(TCHEM_ATM_ENABLE_TEST OFF CACHE BOOL "Disable TChem-atm tests within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_EXAMPLE OFF CACHE BOOL "Disable TChem-atm examples within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_SKYWALKER OFF CACHE BOOL "Disable TChem-atm skywalker within E3SM" FORCE)
# Enable Sacado-based analytical Jacobian for atmospheric chemistry (UCI/gas-phase)
option(EAMXX_TCHEM_ENABLE_SACADO_JACOBIAN "Enable Sacado analytical Jacobian for TChem-atm atmospheric chemistry" ON)
set(TCHEM_ATM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY ${EAMXX_TCHEM_ENABLE_SACADO_JACOBIAN} CACHE BOOL "Enable Sacado Jacobian in TChem-atm" FORCE)
set(TCHEM_ATM_ENABLE_INSTALL OFF CACHE BOOL "Disable TChem-atm install exports within E3SM" FORCE)
if (SCREAM_DOUBLE_PRECISION)
  set(TCHEM_ATM_ENABLE_REAL_TYPE "double" CACHE STRING "TChem-atm real type" FORCE)
else()
  set(TCHEM_ATM_ENABLE_REAL_TYPE "float" CACHE STRING "TChem-atm real type" FORCE)
endif()
add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/src ${CMAKE_BINARY_DIR}/externals/tchem-atm)

# Add the EAMxx TChem-atm process interface library
add_library(eamxx_tchem_atm
  eamxx_tchem_atm_process_interface.cpp
  eamxx_tchem_atm_tchem_functions.cpp)
target_compile_definitions(eamxx_tchem_atm PUBLIC EAMXX_HAS_TCHEM_ATM)
target_include_directories(eamxx_tchem_atm PRIVATE ${EXTERNALS_SOURCE_DIR}/tchem-atm/src/core)
target_link_libraries(eamxx_tchem_atm PUBLIC tchem_atm yaml-cpp mam4xx mam eamxx_physics_share csm_share scream_share)

# Add Sundials support if enabled
if (EAMXX_TCHEM_ENABLE_SACADO_JACOBIAN)
  target_compile_definitions(eamxx_tchem_atm PUBLIC TCHEM_ATM_ENABLE_SACADO_JACOBIAN_ATMOSPHERIC_CHEMISTRY)
  message(STATUS "TChem-atm: Sacado analytical Jacobian ENABLED for atmospheric chemistry")
else()
  message(STATUS "TChem-atm: Sacado analytical Jacobian DISABLED (using numerical Jacobian)")
endif()
if (EAMXX_TCHEM_ENABLE_SUNDIALS)
  target_compile_definitions(eamxx_tchem_atm PUBLIC TCHEM_ATM_ENABLE_SUNDIALS)
  # Also define TCHEM_ATM_ENABLE_GPU if using GPU execution space
  if (Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP)
    target_compile_definitions(eamxx_tchem_atm PUBLIC TCHEM_ATM_ENABLE_GPU)
  endif()
  # Link against Sundials CVODE and Kokkos-based modules
  if (TARGET sundials_cvode)
    target_link_libraries(eamxx_tchem_atm PUBLIC sundials_cvode)
  elseif (TARGET SUNDIALS::cvode)
    target_link_libraries(eamxx_tchem_atm PUBLIC SUNDIALS::cvode)
  endif()
  if (TARGET sundials_nveckokkos)
    target_link_libraries(eamxx_tchem_atm PUBLIC sundials_nveckokkos)
  endif()
  if (TARGET sundials_sunmatrixkokkosdense)
    target_link_libraries(eamxx_tchem_atm PUBLIC sundials_sunmatrixkokkosdense)
  endif()
  if (TARGET sundials_sunlinsolkokkosdense)
    target_link_libraries(eamxx_tchem_atm PUBLIC sundials_sunlinsolkokkosdense)
  endif()
endif()

if (TARGET eamxx_physics)
  # Add this library to eamxx_physics
  target_link_libraries(eamxx_physics INTERFACE eamxx_tchem_atm)
endif()

if (NOT SCREAM_LIB_ONLY)
  add_subdirectory(tests)
endif()
