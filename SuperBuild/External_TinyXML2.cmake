set(proj TinyXML2)

# Set dependency list
set(${proj}_DEPENDS
  ""
  )

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj)

if(${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj} is not supported !")
endif()

# Sanity checks
if(DEFINED TinyXML2_DIR AND NOT EXISTS ${TinyXML2_DIR})
  message(FATAL_ERROR "TinyXML2_DIR [${TinyXML2_DIR}] variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED ${proj}_DIR AND NOT ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)

  ExternalProject_Add(${proj}
    ${${proj}_EP_ARGS}
    GIT_REPOSITORY "https://github.com/leethomason/tinyxml2.git"
    GIT_TAG "321ea883b7190d4e85cae5512a12e5eaa8f8731f" #10.0.0-20240313
    SOURCE_DIR ${EP_SOURCE_DIR}
    BINARY_DIR ${EP_BINARY_DIR}
    CMAKE_CACHE_ARGS
      -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
      -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
      -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
      -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
      -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
      -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
      -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
      -DBUILD_TESTING:BOOL=OFF
      -Dtinyxml2_SHARED_LIBS:BOOL=ON
      # Output directory
      -DCMAKE_RUNTIME_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_BIN_DIR}
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH=${CMAKE_BINARY_DIR}/${Slicer_THIRDPARTY_LIB_DIR}
      -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}
      -DCMAKE_INSTALL_LIBDIR:PATH=${Slicer_THIRDPARTY_LIB_DIR}
    DEPENDS
      ${${proj}_DEPENDS}
    INSTALL_COMMAND ""
    )
  set(${proj}_DIR ${EP_BINARY_DIR})

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
endif()

mark_as_superbuild(${proj}_DIR:PATH)

# Add a custom target that depends on the external project
add_custom_target(slicer_${proj}_install ALL
    COMMENT "Installing .so  and python files to ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}/lib"
)

add_dependencies(slicer_${proj}_install ${proj})

add_custom_command(
    TARGET slicer_${proj}_install POST_BUILD
    COMMAND ${CMAKE_COMMAND} -D LIB_DIR="${TinyXML2_DIR}" -D CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}" -D EXTENSION_BUILD_SUBDIRECTORY="${EXTENSION_BUILD_SUBDIRECTORY}" -D Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR="${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}" -P ${CMAKE_SOURCE_DIR}/cmake/InstallSOFiles.cmake
    COMMENT "Copying SO files..."
)
