#-----------------------------------------------------------------------------
# External project common settings
#-----------------------------------------------------------------------------

set(ep_common_c_flags "${CMAKE_C_FLAGS_INIT} ${ADDITIONAL_C_FLAGS}")
set(ep_common_cxx_flags "${CMAKE_CXX_FLAGS_INIT} ${ADDITIONAL_CXX_FLAGS}")

#-----------------------------------------------------------------------------
# Top-level "external" project
#-----------------------------------------------------------------------------

# Extension dependencies
foreach(dep ${EXTENSION_DEPENDS})
  mark_as_superbuild(${dep}_DIR)
endforeach()

set(proj ${SUPERBUILD_TOPLEVEL_PROJECT})

# Project dependencies
set(${proj}_DEPENDS
  SOFA
   )

ExternalProject_Include_Dependencies(${proj}
  PROJECT_VAR proj
  SUPERBUILD_VAR ${EXTENSION_NAME}_SUPERBUILD
  )

ExternalProject_Add(${proj}
  ${${proj}_EP_ARGS}
  DOWNLOAD_COMMAND ""
  INSTALL_COMMAND ""
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  BINARY_DIR ${EXTENSION_BUILD_SUBDIRECTORY}
  CMAKE_CACHE_ARGS
    # Compiler settings
    -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
    -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
    -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
    -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
    -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
    -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
    -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
    # Output directories
    -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
    # Superbuild
    -D${EXTENSION_NAME}_SUPERBUILD:BOOL=OFF
    -DEXTENSION_SUPERBUILD_BINARY_DIR:PATH=${${EXTENSION_NAME}_BINARY_DIR}
  DEPENDS
    ${${proj}_DEPENDS}
  )

ExternalProject_AlwaysConfigure(${proj})

# # Add a custom target that depends on the external project
# add_custom_target(install_so_files ALL
#     COMMENT "Installing .so files to ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}"
# )

# # Add dependencies to ensure this target is built after the external project
# add_dependencies(install_so_files ${proj})

# # Command to create the installation directory (if it does not exist)
# add_custom_command(TARGET install_so_files PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}
# )

# # Installation of SOFA files
# file(GLOB_RECURSE SO_FILES "${SOFA_DIR}/lib/*.so*")
# foreach(SO_FILE IN LISTS SO_FILES)
#     add_custom_command(TARGET install_so_files POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SO_FILE} ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}
#         COMMENT "Copying ${SO_FILE} to ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}"
#     )
# endforeach()

# file(GLOB_RECURSE SO_FILES "${TinyXML2_DIR}/lib/*.so*")
# foreach(SO_FILE IN LISTS SO_FILES)
#     add_custom_command(TARGET install_so_files POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SO_FILE} ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}
#         COMMENT "Copying ${SO_FILE} to ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}"
#     )
# endforeach()

# # Create the destination directory if it doesn't exist
# add_custom_command(TARGET install_so_files PRE_BUILD
#     COMMAND ${CMAKE_COMMAND} -E make_directory  "${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}/Python"
# )

# set(SOFA_PYTHON_DIR ${SOFA_DIR}/lib/python3/site-packages)
# # Get all subdirectories within the sofa python3 directory
# file(GLOB CHILDREN RELATIVE ${SOFA_PYTHON_DIR} ${SOFA_PYTHON_DIR}/*)
# foreach(child ${CHILDREN})
#     if(IS_DIRECTORY ${SOFA_PYTHON_DIR}/${child})
#         # Define the source subdirectory and the corresponding destination
#         set(SOURCE_SUBDIR ${SOFA_PYTHON_DIR}/${child})
#         set(DEST_SUBDIR  ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}/Python/${child})

#         # Copy the subdirectory
#         add_custom_command(TARGET install_so_files POST_BUILD
#             COMMAND ${CMAKE_COMMAND} -E echo "Copying ${SOURCE_SUBDIR} to ${DEST_SUBDIR}"
#             COMMAND ${CMAKE_COMMAND} -E copy_directory ${SOURCE_SUBDIR} ${DEST_SUBDIR}
#             COMMENT "Copying subdirectory ${child} to ${DEST_SUBDIR}"
#         )
#     endif()
# endforeach()
