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

# TChem-atm external package
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

# Add the EAMxx TChem-atm interface library
add_library(eamxx_tchem_atm INTERFACE)
target_link_libraries(eamxx_tchem_atm INTERFACE tchem_atm)

if (TARGET eamxx_physics)
  # Add this library to eamxx_physics
  target_link_libraries(eamxx_physics INTERFACE eamxx_tchem_atm)
endif()
