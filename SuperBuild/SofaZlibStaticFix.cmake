# Injected into SOFA's configure via CMAKE_PROJECT_Sofa_INCLUDE.
#
# Slicer ships zlib as a STATIC library and installs no DLL.  On Windows
# CMake's FindZLIB sees the .lib, assumes it is an IMPORT library for a
# DLL, and creates ZLIB::ZLIB as a SHARED imported target: IMPORTED_IMPLIB
# is set to the .lib and IMPORTED_LOCATION -- which for a shared imported
# target means the DLL -- is left EMPTY.
#
# SOFA's sofa_install_libraries() then asks each target for
# LOCATION_<CONFIG> (SofaMacrosInstall.cmake), which resolves to that
# empty DLL path, and the configure dies with
#
#   IMPORTED_LOCATION or IMPORTED_IMPLIB not set for imported target
#   "ZLIB::ZLIB" configuration "RELEASE".
#
# Linux never sees this: with no import-library concept the target is
# created as UNKNOWN with IMPORTED_LOCATION pointing straight at
# libzlib.a, so LOCATION_<CONFIG> resolves.  Note the difference is the
# target TYPE, not static-vs-shared -- Slicer ships zlib static on both
# platforms.
#
# Declaring the target ourselves, correctly, before SOFA looks for it
# makes find_package(ZLIB) reuse it instead of inventing a shared one.
# Setting ZLIB_USE_STATIC_LIBS does NOT work here: ZLIB_LIBRARY is
# already in the cache, so FindZLIB short-circuits before reaching its
# static-library branch (tested).
if(WIN32 AND NOT TARGET ZLIB::ZLIB AND ZLIB_LIBRARY AND ZLIB_INCLUDE_DIR)
  add_library(ZLIB::ZLIB STATIC IMPORTED GLOBAL)
  set_target_properties(ZLIB::ZLIB PROPERTIES
    IMPORTED_LOCATION "${ZLIB_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}")
  set(ZLIB_FOUND TRUE)
  set(ZLIB_LIBRARIES "${ZLIB_LIBRARY}")
  set(ZLIB_INCLUDE_DIRS "${ZLIB_INCLUDE_DIR}")
  message(STATUS "Slicer-SOFA: pre-declared ZLIB::ZLIB as STATIC (Slicer ships no zlib DLL)")
endif()
