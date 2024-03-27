# Use variables passed via -D for paths
# Iterate over subdirectories in the specified Python directory and copy them

# Assume LIB_PYTHON_DIR and other variables are passed to the script
file(GLOB CHILDREN RELATIVE ${LIB_PYTHON_DIR} ${LIB_PYTHON_DIR}/*)
foreach(child ${CHILDREN})
    if(IS_DIRECTORY ${LIB_PYTHON_DIR}/${child})
        # Compute source and destination paths
        set(SOURCE_SUBDIR ${LIB_PYTHON_DIR}/${child})
        set(DEST_SUBDIR  ${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}/Python)

        # Perform the copy
        file(COPY ${SOURCE_SUBDIR} DESTINATION ${DEST_SUBDIR})
    endif()
endforeach()
