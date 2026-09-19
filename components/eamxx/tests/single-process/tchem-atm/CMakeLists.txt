include (ScreamUtils)

# ===================== Default TChem-atm test (implicit_euler) =====================
set (TEST_BASE_NAME tchem_atm_standalone)

# Use a checked-in TChem-atm chemistry mechanism file for this standalone smoke test.
#set (TCHEM_ATM_CHEM_FILE
#     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../externals/tchem-atm/src/examples/runs/atmospheric_chemistry/uci_col/uci_v2_test3.yaml")
set (TCHEM_ATM_CHEM_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../externals/tchem-atm/src/examples/runs/atmospheric_chemistry/uci_col/uci_v2_test3_gas.yaml")

# Create the test
CreateADUnitTest(${TEST_BASE_NAME}
  LABELS tchem_atm physics
  LIBS eamxx_tchem_atm
  MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
)

# Set AD configurable options
set (ATM_TIME_STEP 300)
SetVarDependingOnTestSize(NUM_STEPS 2 8 10)
set (RUN_T0 2021-10-12-45000)

# Copy (and configure) yaml files needed by tests
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
  LABELS tchem_atm physics
  META_FIXTURES_REQUIRED ${FIXTURES_BASE_NAME}_npMPIRANKS_omp1
)

if (SCREAM_ENABLE_BASELINE_TESTS)
  # Compare one of the output files with the baselines.
  # Note: one is enough, since we already check that np1 is BFB with npX
  set (OUT_FILE ${TEST_BASE_NAME}_output.INSTANT.nsteps_x1.np${TEST_RANK_END}.${RUN_T0}.nc)
  CreateBaselineTest(${TEST_BASE_NAME} ${TEST_RANK_END} ${OUT_FILE} ${FIXTURES_BASE_NAME})
endif()

# ===================== CVODE batch solver test =====================
# Only build this test if SUNDIALS is enabled
if (EAMXX_TCHEM_ENABLE_SUNDIALS)
  set (TEST_BASE_NAME_CVODE tchem_atm_cvode_standalone)

 # Create the CVODE test
 CreateADUnitTest(${TEST_BASE_NAME_CVODE}
   LABELS tchem_atm physics cvode
   LIBS eamxx_tchem_atm
   MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
   EXE_ARGS "--args --ifile input_cvode.yaml"
 )

  # Copy (and configure) yaml files needed by CVODE test
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input_cvode.yaml
                 ${CMAKE_CURRENT_BINARY_DIR}/input_cvode.yaml)

  # Compare output files produced by npX tests, to ensure they are bfb
  CompareNCFilesFamilyMpi (
    TEST_BASE_NAME ${TEST_BASE_NAME_CVODE}
    FILE_META_NAME ${TEST_BASE_NAME_CVODE}_output.INSTANT.nsteps_x1.npMPIRANKS.${RUN_T0}.nc
    MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
    LABELS tchem_atm physics cvode
    META_FIXTURES_REQUIRED ${TEST_BASE_NAME_CVODE}_npMPIRANKS_omp1
  )

  if (SCREAM_ENABLE_BASELINE_TESTS)
    set (OUT_FILE_CVODE ${TEST_BASE_NAME_CVODE}_output.INSTANT.nsteps_x1.np${TEST_RANK_END}.${RUN_T0}.nc)
    CreateBaselineTest(${TEST_BASE_NAME_CVODE} ${TEST_RANK_END} ${OUT_FILE_CVODE} ${TEST_BASE_NAME_CVODE})
  endif()
endif()
