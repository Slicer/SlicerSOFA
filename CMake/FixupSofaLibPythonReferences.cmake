# Point SofaPython3's bindings at Slicer's libpython (macOS packaging).
#
# SofaPython3 links libpython through the path Slicer's *build tree* exposes,
# @rpath/lib/Python/lib/libpython3.X.dylib.  In an installed Slicer that library
# lives at lib/libpython3.X.dylib -- Slicer's own binaries reference
# @rpath/lib/libpython3.X.dylib and nothing in the bundle references the old
# location -- so no rpath resolves it, dlopen of Sofa.Helper fails, and
# "import Sofa" raises.  Every module of this extension is unusable as a result.
#
# Slicer's extension fixup does not rewrite this reference, and it cannot be
# corrected from an install() rule: for a SuperBuild extension the fixup script
# is appended to CPACK_INSTALL_CMAKE_PROJECTS last, so it always runs after our
# install rules.  This script therefore runs from CPACK_PRE_BUILD_SCRIPTS, which
# CPack executes once the staging tree is fully installed and fixed up, but
# before the package is produced.

if(NOT APPLE)
  return()
endif()

set(_staging_directory "${CPACK_TEMPORARY_DIRECTORY}")
if(NOT _staging_directory)
  set(_staging_directory "${CPACK_TEMPORARY_INSTALL_DIRECTORY}")
endif()
if(NOT _staging_directory OR NOT EXISTS "${_staging_directory}")
  message(WARNING "Fixup of libpython references skipped: no CPack staging directory")
  return()
endif()

file(GLOB_RECURSE _binaries "${_staging_directory}/*.so" "${_staging_directory}/*.dylib")

set(_fixed_count 0)
foreach(_binary IN LISTS _binaries)
  execute_process(
    COMMAND otool -L "${_binary}"
    OUTPUT_VARIABLE _dependencies
    ERROR_QUIET
    )
  string(REGEX MATCH "@rpath/lib/Python/lib/(libpython[0-9.]*[.]dylib)"
    _stale_reference "${_dependencies}")
  if(_stale_reference)
    execute_process(
      COMMAND install_name_tool -change
        "${_stale_reference}" "@rpath/lib/${CMAKE_MATCH_1}" "${_binary}"
      RESULT_VARIABLE _result
      ERROR_VARIABLE _error
      )
    if(NOT _result EQUAL 0)
      message(WARNING "Could not rewrite libpython reference in ${_binary}: ${_error}")
      continue()
    endif()
    # Editing a load command invalidates the signature; re-sign ad hoc.
    execute_process(COMMAND codesign --force --sign - "${_binary}" ERROR_QUIET)
    math(EXPR _fixed_count "${_fixed_count} + 1")
  endif()
endforeach()

message(STATUS "Fixed the libpython reference of ${_fixed_count} SOFA binaries")
