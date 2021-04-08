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
        template <typename TQueue>
        class xilinx_noop_kernel; // Workaround for XRT issues. See description in the c'tor of the Xilinx queues.
    }
}

#endif
