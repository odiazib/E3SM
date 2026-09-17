include(${CMAKE_CURRENT_LIST_DIR}/common.cmake)
common_setup()

# Flight is a Sandia SNL cluster with Intel compilers and OpenMPI

# Input data location
set(SCREAM_INPUT_ROOT "/projects/ccsm/inputdata" CACHE STRING "")

# Intel compiler-specific flags
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  # Disable some Intel compiler remarks that are verbose
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=remark" CACHE STRING "" FORCE)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -diag-disable=remark" CACHE STRING "" FORCE)
endif()

# Use Intel MKL for BLAS/LAPACK
set(SCREAM_LINK_OPTIONS "-mkl" CACHE STRING "")

# OpenMPI settings
set(EKAT_MPI_NP_FLAG "-np" CACHE STRING "")
