/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <sycl/sycl.hpp>

#include <algorithm>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace alpaka
{
    namespace traits
    {
        namespace detail
        {
            template <typename T, typename = void>
            struct is_sycl_enqueueable : std::false_type {};

            template <typename T>
            struct is_sycl_enqueueable<T, std::void_t<decltype(T::is_sycl_enqueueable)>>
            : std::true_type
            {
            };

            inline auto remove_completed(std::vector<sycl::event>& events)
            {
                using namespace sycl;

                std::remove_if(begin(events), end(events), [](event const& ev)
                {
                    return (ev.get_info<info::event::command_execution_status>()
                                == info::event_command_status::complete);
                });
            }

        }
    }

    namespace detail
    {
#if defined(ALPAKA_SYCL_BACKEND_XILINX)
        // This is a workaround for a known bug in the Xilinx SYCL implementation:
        // https://github.com/triSYCL/sycl/issues/40. Unless we jumpstart the runtime by executing a NOOP kernel,
        // copies / memsets preceding the first kernel launch will fail because XRT fails to map them to the
        // virtual devices in sw_emu / hw_emu mode.
        class alpaka_xilinx_noop_kernel;

        inline auto jumpstart_device(sycl::queue& queue)
        {
            static auto started = false;
            if(!started)
            {
                // We align the data to 4KiB to prevent XRT's warnings about unaligned memory.
                constexpr auto size = 10;
                auto data = std::shared_ptr<int[]>{new(std::align_val_t{4096}) int[size]};

                // Another issue: All kernels must have at least 1 accessor or v++ will complain.
                auto buf = sycl::buffer<int, 1>{data, sycl::range<1>{size}};

                queue.submit([&](sycl::handler& cgh)
                {
                    auto acc = buf.get_access<sycl::access::mode::write>(cgh);
                    cgh.single_task<alpaka_xilinx_noop_kernel>([=]()
                    {
                        acc[0] = 1;
                    });
                });
                queue.wait_and_throw();

                started = true;
            }
        }
#endif
    }
}

#endif
