set(proj Boost)

# Set dependency list
set(${proj}_DEPENDS
  ""
)

# Include dependent projects if any
ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj)

if(${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})
  message(FATAL_ERROR "Enabling ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj} is not supported!")
endif()

# Sanity checks
if(DEFINED Boost_DIR AND NOT EXISTS ${Boost_DIR})
  message(FATAL_ERROR "Boost_DIR [${Boost_DIR}] variable is defined but corresponds to nonexistent directory")
endif()

if(NOT DEFINED ${proj}_DIR AND NOT ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})

  set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
  set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
  set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)
  set(BOOST_URL "https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.gz")
  set(BOOST_URL_HASH "SHA256=4d27e9efed0f6f152dc28db6430b9d3dfb40c0345da7342eaa5a987dde57bd95") # Replace <expected hash value> with the actual SHA256 hash of the tar.gz file

  if(WIN32)
    set(BUILD_COMMAND ${EP_SOURCE_DIR}/bootstrap.bat)
  else()
    set(BUILD_COMMAND ${EP_SOURCE_DIR}/bootstrap.sh)
  endif()

  set(EXTERNAL_PROJECT_OPTIONAL_ARGS)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    list(APPEND EXTERNAL_PROJECT_OPTIONAL_ARGS
      DOWNLOAD_EXTRACT_TIMESTAMP 1
      )
  endif()

  ExternalProject_Add(${proj}
    ${EXTERNAL_PROJECT_OPTIONAL_ARGS}
    URL ${BOOST_URL}
    URL_HASH ${BOOST_URL_HASH}
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/download
    SOURCE_DIR ${EP_SOURCE_DIR}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${BUILD_COMMAND}
    INSTALL_COMMAND ${EP_SOURCE_DIR}/b2 --prefix=${EP_INSTALL_DIR} install
    BUILD_IN_SOURCE 1
    DEPENDS
      ${${proj}_DEPENDS}
  )
  set(${proj}_DIR ${EP_INSTALL_DIR})

else()
  ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
endif()

mark_as_superbuild(${proj}_DIR:PATH)


# set(proj Boost)

# # Set dependency list
# set(${proj}_DEPENDS
#   ""
#   )

# # Include dependent projects if any
# ExternalProject_Include_Dependencies(${proj} PROJECT_VAR proj)

# if(${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})
#   message(FATAL_ERROR "Enabling ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj} is not supported !")
# endif()

# # Sanity checks
# if(DEFINED Boost_DIR AND NOT EXISTS ${Boost_DIR})
#   message(FATAL_ERROR "Boost_DIR [${Boost_DIR}] variable is defined but corresponds to nonexistent directory")
# endif()

# if(NOT DEFINED ${proj}_DIR AND NOT ${SUPERBUILD_TOPLEVEL_PROJECT}_USE_SYSTEM_${proj})

#   set(EP_SOURCE_DIR ${CMAKE_BINARY_DIR}/${proj})
#   set(EP_BINARY_DIR ${CMAKE_BINARY_DIR}/${proj}-build)
#   set(EP_INSTALL_DIR ${CMAKE_BINARY_DIR}/${proj}-install)

#   ExternalProject_Add(${proj}
#     ${${proj}_EP_ARGS}
#     GIT_REPOSITORY "https://github.com/boostorg/boost.git"
#     GIT_TAG "ad09f667e61e18f5c31590941e748ac38e5a81bf" #boost-1.84.0-20240313
#     SOURCE_DIR ${EP_SOURCE_DIR}
#     BINARY_DIR ${EP_BINARY_DIR}
#     INSTALL_DIR ${EP_INSTALL_DIR}
#     CMAKE_CACHE_ARGS
#       -DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}
#       -DCMAKE_C_FLAGS:STRING=${ep_common_c_flags}
#       -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}
#       -DCMAKE_CXX_FLAGS:STRING=${ep_common_cxx_flags}
#       -DCMAKE_CXX_STANDARD:STRING=${CMAKE_CXX_STANDARD}
#       -DCMAKE_CXX_STANDARD_REQUIRED:BOOL=${CMAKE_CXX_STANDARD_REQUIRED}
#       -DCMAKE_CXX_EXTENSIONS:BOOL=${CMAKE_CXX_EXTENSIONS}
#       -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
#       -DBOOST_ROOT:PATH=${EP_SOURCE_DIR}
#       -DBOOST_INCLUDEDIR:PATH=${EP_SOURCE_DIR}
#       -DBOOST_LIBRARYDIR:PATH=${EP_BINARY_DIR}/stage/lib
#       -DBUILD_TESTING:BOOL=OFF
#     DEPENDS
#       ${${proj}_DEPENDS}
#     )
#   set(${proj}_DIR ${EP_INSTALL_DIR})

# else()
#   ExternalProject_Add_Empty(${proj} DEPENDS ${${proj}_DEPENDS})
# endif()

# mark_as_superbuild(${proj}_DIR:PATH)
