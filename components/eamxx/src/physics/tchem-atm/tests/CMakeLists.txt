include(ScreamUtils)
set(TCHEM_TESTS_SRCS
  tchem_photo_test.cpp
) # TCHEM_TESTS_SRCS

set(BASELINE_FILE_ARG "-n") # no baselines

CreateUnitTest(tchem_atm_tests
  SOURCES ${TCHEM_TESTS_SRCS}
  LIBS eamxx_tchem_atm eamxx_io
  EXE_ARGS "--args ${BASELINE_FILE_ARG}"
  THREADS 1
  LABELS "tchem-atm;physics")