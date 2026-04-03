include (ScreamUtils)

set (TEST_BASE_NAME tchem_atm_standalone)

# Use a checked-in TChem-atm chemistry mechanism file for this standalone smoke test.
set (TCHEM_ATM_CHEM_FILE
     "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../externals/tchem-atm/src/examples/runs/atmospheric_chemistry/uci_e3sm_v3/uci_v2_gas.yaml")

# Create the test
CreateADUnitTest(${TEST_BASE_NAME}
  LABELS tchem_atm physics
  LIBS eamxx_tchem_atm
  MPI_RANKS ${TEST_RANK_START} ${TEST_RANK_END}
)

# Set AD configurable options
set (ATM_TIME_STEP 300)
SetVarDependingOnTestSize(NUM_STEPS 1 2 4)
set (RUN_T0 2021-10-12-45000)

# Copy (and configure) yaml files needed by tests
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input.yaml
               ${CMAKE_CURRENT_BINARY_DIR}/input.yaml)

# Ensure test input files are present in the data dir
GetInputFile(scream/init/${EAMxx_tests_IC_FILE_72lev})
GetInputFile(cam/topo/${EAMxx_tests_TOPO_FILE})
