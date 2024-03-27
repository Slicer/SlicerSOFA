# Assuming LIB_DIR, EXTENSION_BUILD_SUBDIRECTORY, and Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR are passed via -D arguments

# Glob the .so files
file(GLOB_RECURSE SO_FILES "${LIB_DIR}/*.so*")

# Iterate over each file and copy it to the desired directory
foreach(SO_FILE IN LISTS SO_FILES)
    file(COPY ${SO_FILE} DESTINATION "${CMAKE_BINARY_DIR}/${EXTENSION_BUILD_SUBDIRECTORY}/${Slicer_INSTALL_QTLOADABLEMODULES_LIB_DIR}/lib")
endforeach()
