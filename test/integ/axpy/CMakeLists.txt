#
# Copyright 2014-2020 Benjamin Worpitz, Axel Huebl, Jan Stephan
#
# This file is part of alpaka.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#

set(_TARGET_NAME "axpy")

append_recursive_files_add_to_src_group("src/" "src/" "cpp" _FILES_SOURCE)

alpaka_add_executable(
    ${_TARGET_NAME}
    ${_FILES_SOURCE})
if(ALPAKA_ACC_SYCL_ENABLE AND ALPAKA_SYCL_ONEAPI_FPGA)
    target_link_libraries(${_TARGET_NAME} PRIVATE $<TARGET_FILE:common_intelfpga>)
    add_dependencies(${_TARGET_NAME} common_intelfpga)
else()
    target_link_libraries(${_TARGET_NAME} PRIVATE common)
endif()

set_target_properties(${_TARGET_NAME} PROPERTIES FOLDER "test/integ")

add_test(NAME ${_TARGET_NAME} COMMAND ${_TARGET_NAME} ${_ALPAKA_TEST_OPTIONS})
