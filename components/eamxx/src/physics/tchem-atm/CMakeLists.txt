if (SCREAM_CIME_BUILD)
  # PROJECT_SOURCE_DIR is SCREAM_ROOT/components
  set(EXTERNALS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../externals")
else()
  # PROJECT_SOURCE_DIR is SCREAM_ROOT/components/eamxx
  set(EXTERNALS_SOURCE_DIR "${PROJECT_SOURCE_DIR}/../../externals")
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
# if (NOT TARGET SUNDIALS::cvode)
#   set(ENABLE_KOKKOS ON CACHE BOOL "Enable Sundials Kokkos support within E3SM" FORCE)
#   set(Kokkos_DIR "${CMAKE_BINARY_DIR}/externals/kokkos" CACHE PATH "Path to built Kokkos package config" FORCE)
#   set(BUILD_ARKODE OFF CACHE BOOL "Disable Sundials ARKODE within E3SM" FORCE)
#   set(BUILD_CVODE ON CACHE BOOL "Enable Sundials CVODE within E3SM" FORCE)
#   set(BUILD_CVODES OFF CACHE BOOL "Disable Sundials CVODES within E3SM" FORCE)
#   set(BUILD_IDA OFF CACHE BOOL "Disable Sundials IDA within E3SM" FORCE)
#   set(BUILD_IDAS OFF CACHE BOOL "Disable Sundials IDAS within E3SM" FORCE)
#   set(BUILD_KINSOL OFF CACHE BOOL "Disable Sundials KINSOL within E3SM" FORCE)
#   set(EXAMPLES_ENABLE_C OFF CACHE BOOL "Disable Sundials C examples within E3SM" FORCE)
#   set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "Disable Sundials CXX examples within E3SM" FORCE)
#   set(EXAMPLES_ENABLE_CUDA OFF CACHE BOOL "Disable Sundials CUDA examples within E3SM" FORCE)
#   set(EXAMPLES_ENABLE_F2003 OFF CACHE BOOL "Disable Sundials F2003 examples within E3SM" FORCE)
#   set(EXAMPLES_INSTALL OFF CACHE BOOL "Disable Sundials example install within E3SM" FORCE)
#   set(SUNDIALS_TEST_ENABLE_DEV_TESTS OFF CACHE BOOL "Disable Sundials dev tests within E3SM" FORCE)
#   set(SUNDIALS_TEST_ENABLE_UNIT_TESTS OFF CACHE BOOL "Disable Sundials unit tests within E3SM" FORCE)
#   set(BUILD_BENCHMARKS OFF CACHE BOOL "Disable Sundials benchmarks within E3SM" FORCE)
#   add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/external/Sundials
#                    ${CMAKE_BINARY_DIR}/externals/tchem-atm/external/Sundials)
# endif()

# TChem-atm external package
# if (NOT SUNDIALS_INSTALL_PATH)
#   set(SUNDIALS_INSTALL_PATH "/pscratch/sd/o/odiazib/e3sm-tchem/TChem-atm/scripts/CUDA/install/sundials" CACHE PATH "Path to Sundials installation")
# endif()
set(TCHEM_ATM_ENABLE_SUNDIALS OFF CACHE BOOL "Disable TChem-atm tests within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_TEST OFF CACHE BOOL "Disable TChem-atm tests within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_EXAMPLE OFF CACHE BOOL "Disable TChem-atm examples within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_SKYWALKER OFF CACHE BOOL "Disable TChem-atm skywalker within E3SM" FORCE)
set(TCHEM_ATM_ENABLE_INSTALL OFF CACHE BOOL "Disable TChem-atm install exports within E3SM" FORCE)
if (SCREAM_DOUBLE_PRECISION)
  set(TCHEM_ATM_ENABLE_REAL_TYPE "double" CACHE STRING "TChem-atm real type" FORCE)
else()
  set(TCHEM_ATM_ENABLE_REAL_TYPE "float" CACHE STRING "TChem-atm real type" FORCE)
endif()
add_subdirectory(${EXTERNALS_SOURCE_DIR}/tchem-atm/src ${CMAKE_BINARY_DIR}/externals/tchem-atm)

# Add the EAMxx TChem-atm process interface library
add_library(eamxx_tchem_atm eamxx_tchem_atm_process_interface.cpp)
target_compile_definitions(eamxx_tchem_atm PUBLIC EAMXX_HAS_TCHEM_ATM)
target_include_directories(eamxx_tchem_atm PRIVATE ${EXTERNALS_SOURCE_DIR}/tchem-atm/src/core)
target_link_libraries(eamxx_tchem_atm PUBLIC tchem_atm eamxx_physics_share csm_share scream_share)

if (TARGET eamxx_physics)
  # Add this library to eamxx_physics
  target_link_libraries(eamxx_physics INTERFACE eamxx_tchem_atm)
endif()
