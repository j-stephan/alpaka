#
# Copyright 2014-2020 Benjamin Worpitz, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

cmake_minimum_required(VERSION 3.15)

set(_COMMON_TARGET_NAME "common")
set(_COMMON_COMPILE_OPTIONS_FILE "devCompileOptions.cmake")
set(_COMMON_INCLUDE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/include")

# Add all the source files in all recursive subdirectories and group them accordingly.
append_recursive_files_add_to_src_group("${_COMMON_INCLUDE_DIRECTORY}" "${_COMMON_INCLUDE_DIRECTORY}" "hpp" _COMMON_FILES_HEADER)

if(ALPAKA_ACC_SYCL_ENABLE AND ALPAKA_SYCL_ONEAPI_FPGA)
    # Intel FPGAs require a special way of static library creation
    set(_COMMON_OBJ_TARGET_NAME "${_COMMON_TARGET_NAME}_obj")
    set(_COMMON_TARGET_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}${_COMMON_TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_COMMON_FPGA_TARGET_NAME "${_COMMON_TARGET_NAME}_intelfpga")

    # Step 1: Create an object library
    add_library(${_COMMON_OBJ_TARGET_NAME} OBJECT ${_COMMON_FILES_HEADER} ${_COMMON_FILES_SOURCE} "devCompileOptions.cmake")
    target_link_libraries(${_COMMON_OBJ_TARGET_NAME} PRIVATE alpaka::alpaka)
    target_include_directories(${_COMMON_OBJ_TARGET_NAME} PRIVATE ${_COMMON_INCLUDE_DIRECTORY})
    target_compile_options(${_COMMON_OBJ_TARGET_NAME} PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})
    set_target_properties(${_COMMON_OBJ_TARGET_NAME} PROPERTIES FOLDER "test")
    target_link_libraries(${_COMMON_OBJ_TARGET_NAME} PRIVATE CatchMain)

    # Step 2: Generate a special FPGA static library
    add_custom_command(OUTPUT ${_COMMON_TARGET_LIB}
	               COMMAND fpga_libtool $<TARGET_OBJECTS:${_COMMON_OBJ_TARGET_NAME}> --target sycl --create ${_COMMON_TARGET_LIB}
		       DEPENDS $<TARGET_OBJECTS:${_COMMON_OBJ_TARGET_NAME}>
		       COMMENT "Packaging FPGA static library ${_COMMON_TARGET_LIB}")
    add_custom_target(${_COMMON_FPGA_TARGET_NAME} DEPENDS ${_COMMON_TARGET_LIB})
    add_dependencies(${_COMMON_FPGA_TARGET_NAME} ${_COMMON_OBJ_TARGET_NAME})

    # Step 3: Create an imported library for easy usage
    add_library(${_COMMON_TARGET_NAME} STATIC IMPORTED GLOBAL)
    set_property(TARGET ${_COMMON_TARGET_NAME} PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${_COMMON_TARGET_LIB}")
    add_dependencies(${_COMMON_TARGET_NAME} ${_COMMON_FPGA_TARGET_NAME})

    target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE alpaka::alpaka)

    target_include_directories(${_COMMON_TARGET_NAME} INTERFACE ${_COMMON_INCLUDE_DIRECTORY})

    include("devCompileOptions.cmake")
    target_compile_options(${_COMMON_TARGET_NAME} INTERFACE ${ALPAKA_DEV_COMPILE_OPTIONS})

    target_link_libraries(${_COMMON_TARGET_NAME} INTERFACE CatchMain)
else()
    add_library(${_COMMON_TARGET_NAME} STATIC ${_COMMON_FILES_HEADER} ${_COMMON_FILES_SOURCE} "devCompileOptions.cmake")

    target_link_libraries(${_COMMON_TARGET_NAME} PUBLIC alpaka::alpaka)

    target_include_directories(${_COMMON_TARGET_NAME} PUBLIC ${_COMMON_INCLUDE_DIRECTORY})

    include("devCompileOptions.cmake")
    target_compile_options(${_COMMON_TARGET_NAME} PUBLIC ${ALPAKA_DEV_COMPILE_OPTIONS})

    if(MSVC)
        target_compile_options(${_COMMON_TARGET_NAME} PUBLIC "/wd4996") # This function or variable may be unsafe. Consider using <safe_version> instead.
        target_compile_options(${_COMMON_TARGET_NAME} PUBLIC "/bigobj")
    endif()

    if(ALPAKA_ACC_GPU_CUDA_ENABLE OR (ALPAKA_ACC_GPU_HIP_ENABLE AND HIP_PLATFORM MATCHES "nvcc"))
        # CUDA driver API is used by EventHostManualTrigger
	target_link_libraries(${_COMMON_TARGET_NAME} PUBLIC "${CUDA_CUDA_LIBRARY}")
	target_compile_definitions(${_COMMON_TARGET_NAME} PUBLIC "CUDA_API_PER_THREAD_DEFAULT_STREAM")
    endif()

    target_link_libraries(${_COMMON_TARGET_NAME} PUBLIC CatchMain)
endif()

set_target_properties(${_COMMON_TARGET_NAME} PROPERTIES FOLDER "test")
