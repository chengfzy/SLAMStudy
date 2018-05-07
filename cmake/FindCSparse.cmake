# Look for csparse; note the difference in the directory specifications!

# find package csparse
# Once done this will define
#  CSPARSE_FOUND - if system found Sophus library
#  CSPARSE_INCLUDE_DIRS/CSPARSE_INCLUDE_DIR - The Sophus include directories
#  CSPARSE_LIBRARIES/CSPARSE_LIBRARY - The libraries needed to use Sophus
#  CSPARSE_DEFINITIONS - Compiler switches required for using Sophus

FIND_PATH(CSPARSE_INCLUDE_DIR NAMES cs.h
  PATHS
  /usr/include/suitesparse
  /usr/include
  /opt/local/include
  /usr/local/include
  /sw/include
  /usr/include/ufsparse
  /opt/local/include/ufsparse
  /usr/local/include/ufsparse
  /sw/include/ufsparse
  DOC "CSParse include directory"
  )

# only find shared library
FIND_LIBRARY(CSPARSE_LIBRARY NAMES libcxsparse.so
  PATHS
  /usr/lib
  /usr/local/lib
  /opt/local/lib
  /sw/lib
  DOC "CSparse library directory"
  )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSPARSE DEFAULT_MSG CSPARSE_INCLUDE_DIR CSPARSE_LIBRARY)

if (CSPARSE_FOUND)
    set(CSPARSE_INCLUDE_DIRS ${CSPARSE_INCLUDE_DIR})
    set(CSPARSE_LIBRARIES ${CSPARSE_LIBRARY})
    set(CSPARSE_DEFINITIONS)
    #message(STATUS "Found CSPARSE libs:" ${CSPARSE_LIBRARY})
endif()

mark_as_advanced(CSPARSE_INCLUDE_DIR CSPARSE_LIBRARY)
