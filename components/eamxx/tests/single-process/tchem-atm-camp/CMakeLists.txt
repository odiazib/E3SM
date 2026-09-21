include (ScreamUtils)

set (TEST_BASE_NAME tchem_atm_camp_standalone)
set (FIXTURES_BASE_NAME ${TEST_BASE_NAME}_generate_output_nc_files)

# CBO5 mechanism files (checked-in with tchem-atm)
set (TCHEM_ATM_CAMP_CHEM_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../externals/tchem-atm/src/examples/runs/atmospheric_chemistry/CB05CL_AE5_w_simpolSOA/config_full_gas.yaml")
set (TCHEM_ATM_CAMP_AERO_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../externals/tchem-atm/src/examples/runs/atmospheric_chemistry/CB05CL_AE5_w_simpolSOA/mechanism_aero.yaml")

# Create the test
CreateADUnitTest(${TEST_BASE_NAME}
  LABELS tchem_atm_camp physics
  LIBS eamxx_tchem_atm
  MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
  FIXTURES_SETUP_INDIVIDUAL ${FIXTURES_BASE_NAME}
)

# Set AD configurable options
set (ATM_TIME_STEP 300)
SetVarDependingOnTestSize(NUM_STEPS 2 8 10)
set (RUN_T0 2021-10-12-45000)

# Copy (and configure) yaml files needed by the test
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/output.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/output.yaml)

# Ensure test input files are present in the data dir
GetInputFile(scream/init/${EAMxx_tests_IC_FILE_72lev})
GetInputFile(cam/topo/${EAMxx_tests_TOPO_FILE})

# Compare output files produced by npX tests, to ensure they are bfb
include (CompareNCFiles)

CompareNCFilesFamilyMpi (
  TEST_BASE_NAME ${TEST_BASE_NAME}
  FILE_META_NAME ${TEST_BASE_NAME}_output.INSTANT.nsteps_x1.npMPIRANKS.${RUN_T0}.nc
  MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
  LABELS tchem_atm_camp physics
  META_FIXTURES_REQUIRED ${FIXTURES_BASE_NAME}_npMPIRANKS_omp1
)
