/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <CL/sycl.hpp>

#include <algorithm>
#include <vector>

namespace alpaka::traits::detail
{
    inline auto remove_completed(std::vector<cl::sycl::event>& events)
    {
        using namespace cl::sycl;

        std::remove_if(begin(events), end(events), [](event const& ev)
        {
            return (ev.get_info<info::event::command_execution_status>() == info::event_command_status::complete);
        });
    }
}

#endif
