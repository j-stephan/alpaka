/* Copyright 2019 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/queue/QueueSyclAsync.hpp>

namespace alpaka
{
    namespace queue
    {
        // There is no synchronous queue in SYCL and implementing one isn't
        // really necessary. SYCL's queues are async by default. If there are
        // dependencies between kernels SYCL will serialize them for us.
        using QueueSyclSync = QueueSyclAsync;
    }
}

#endif
