include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)
common_setup()

set(EKAT_MACH_FILES_PATH ${CMAKE_CURRENT_LIST_DIR}/../../../../externals/ekat/cmake/machine-files)

# Input data location
set(SCREAM_INPUT_ROOT "/projects/ccsm/inputdata" CACHE STRING "")

# NetCDF and PnetCDF paths
# These are set via environment variables from modules, but we need to pass them to CMake
if (DEFINED ENV{NETCDF_C_PATH})
  set(NetCDF_C_PATH "$ENV{NETCDF_C_PATH}" CACHE STRING "")
endif()
set(NetCDF_Fortran_PATH "/projects/sems/install/boca/acme/manual/netcdf-fortran/4.6.3" CACHE STRING "")
if (DEFINED ENV{PNETCDF_PATH})
  set(PnetCDF_C_PATH "$ENV{PNETCDF_PATH}" CACHE STRING "")
  set(PnetCDF_Fortran_PATH "$ENV{PNETCDF_PATH}" CACHE STRING "")
endif()

# Get AMD arch settings
include(${EKAT_MACH_FILES_PATH}/kokkos/intel-skx.cmake)

# Add OpenMP settings in standalone mode OR e3sm with compile_threaded=ON
if (NOT "${PROJECT_NAME}" STREQUAL "E3SM" OR compile_threaded)
  include(${EKAT_MACH_FILES_PATH}/kokkos/openmp.cmake)
endif()

# Use srun for standalone testing
include(${EKAT_MACH_FILES_PATH}/mpi/srun.cmake)
