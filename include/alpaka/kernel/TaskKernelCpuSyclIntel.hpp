/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)

#include <alpaka/kernel/TaskKernelGenericSycl.hpp>

namespace alpaka
{
    template <typename TDim, typename TIdx>
    class AccCpuSyclIntel;

    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    using TaskKernelCpuSyclIntel = TaskKernelGenericSycl<AccCpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>;
}

#endif
