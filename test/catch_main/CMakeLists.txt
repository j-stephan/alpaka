#
# Copyright 2015-2020 Benjamin Worpitz, Axel Huebl
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

option(ALPAKA_USE_INTERNAL_CATCH2 "Use internally shipped Catch2" ON)

if(ALPAKA_USE_INTERNAL_CATCH2)
    message(STATUS "Catch2: Using INTERNAL version 2.13.3")
else()
    find_package(Catch2 2.13.3 CONFIG REQUIRED)
    set_target_properties(Catch2::Catch2 PROPERTIES IMPORTED_GLOBAL TRUE)
    message(STATUS "Catch2: Found version ${Catch2_VERSION}")
endif()

if(ALPAKA_ACC_SYCL_ENABLE AND ALPAKA_SYCL_ONEAPI_FPGA)
    # Intel FPGAs require a special way of static library creation
    set(_CATCHMAIN_TARGET_NAME "CatchMain")
    set(_CATCHMAIN_OBJ_TARGET_NAME "${_CATCHMAIN_TARGET_NAME}_obj")
    set(_CATCHMAIN_TARGET_LIB "${CMAKE_STATIC_LIBRARY_PREFIX}${_CATCHMAIN_TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_CATCHMAIN_FPGA_TARGET_NAME "${_CATCHMAIN_TARGET_NAME}_intelfpga")
    
    # Step 1: Create an object library
    add_library(${_CATCHMAIN_OBJ_TARGET_NAME} OBJECT src/CatchMain.cpp)

    target_compile_definitions(${_CATCHMAIN_OBJ_TARGET_NAME} PRIVATE "CATCH_CONFIG_FAST_COMPILE")
    if(ALPAKA_USE_INTERNAL_CATCH2)
        target_include_directories(${_CATCHMAIN_OBJ_TARGET_NAME} SYSTEM PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../thirdParty/catch2/include)
    else()
        target_include_directories(${_CATCHMAIN_OBJ_TARGET_NAME} SYSTEM PRIVATE $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    # Step 2: Generate a special FPGA static library
    add_custom_command(OUTPUT ${_CATCHMAIN_TARGET_LIB}
                       COMMAND fpga_libtool $<TARGET_OBJECTS:CatchMain_obj> --target sycl --create ${_CATCHMAIN_TARGET_LIB}
                       DEPENDS $<TARGET_OBJECTS:CatchMain_obj>
                       COMMENT "Packaging FPGA static library ${_CATCHMAIN_TARGET_LIB}")
    add_custom_target(${_CATCHMAIN_FPGA_TARGET_NAME} DEPENDS ${_CATCHMAIN_TARGET_LIB})
    add_dependencies(${_CATCHMAIN_FPGA_TARGET_NAME} ${_CATCHMAIN_OBJ_TARGET_NAME})

    # Step 3: Create an imported library for easy usage
    add_library(${_CATCHMAIN_TARGET_NAME} STATIC IMPORTED GLOBAL)
    set_property(TARGET ${_CATCHMAIN_TARGET_NAME} PROPERTY IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/${_CATCHMAIN_TARGET_LIB}")
    add_dependencies(${_CATCHMAIN_TARGET_NAME} ${_CATCHMAIN_FPGA_TARGET_NAME})

    target_compile_definitions(${_CATCHMAIN_TARGET_NAME} INTERFACE "CATCH_CONFIG_FAST_COMPILE")
    if(ALPAKA_USE_INTERNAL_CATCH2)
        target_include_directories(${_CATCHMAIN_TARGET_NAME} SYSTEM INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../thirdParty/catch2/include)
    else()
        target_include_directories(${_CATCHMAIN_TARGET_NAME} SYSTEM INTERFACE $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)
    endif() 
else()
    add_library(CatchMain src/CatchMain.cpp)
    # target_compile_features(CatchMain PUBLIC cxx_std_14)  # min C++14

    target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_FAST_COMPILE")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        # Workaround for STL atomic issue: https://forums.developer.nvidia.com/t/support-for-atomic-in-libstdc-missing/135403/2
        # still appears in NVHPC 20.7
        target_compile_definitions(CatchMain PUBLIC "__GCC_ATOMIC_TEST_AND_SET_TRUEVAL=1")
    endif()
    if(MSVC)
        target_compile_definitions(CatchMain PUBLIC "CATCH_CONFIG_WINDOWS_CRTDBG")
        target_compile_options(CatchMain PUBLIC "/bigobj")
    endif()

    if(ALPAKA_USE_INTERNAL_CATCH2)
        target_include_directories(CatchMain SYSTEM PUBLIC
            ${CMAKE_CURRENT_LIST_DIR}/../../thirdParty/catch2/include)
    else()
        target_include_directories(CatchMain SYSTEM PUBLIC
            $<TARGET_PROPERTY:Catch2::Catch2,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()
endif()

set_target_properties(CatchMain PROPERTIES
    FOLDER "test"
    CXX_STANDARD 14  # exactly C++14
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
