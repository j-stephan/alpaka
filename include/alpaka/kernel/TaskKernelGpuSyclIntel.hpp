/* Copyright 2020 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/queue/Traits.hpp>

// Implementation details.
#include <alpaka/acc/AccUniformSycl.hpp>
#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/kernel/TaskKernelUniformSycl.hpp>
#include <alpaka/queue/QueueSyclNonBlocking.hpp>
#include <alpaka/queue/QueueSyclBlocking.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <alpaka/acc/Traits.hpp>
    #include <alpaka/dev/Traits.hpp>
    #include <alpaka/workdiv/WorkDivHelpers.hpp>
#endif

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/meta/ApplyTuple.hpp>
#include <alpaka/meta/Metafunctions.hpp>

#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_MINIMAL
    #include <iostream>
#endif

#include <stl-tuple/STLTuple.hpp> // computecpp-sdk

namespace alpaka
{
    //#############################################################################
    //! The SYCL accelerator execution task.
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelGpuSyclIntel final : public detail::TaskKernelSyclImpl<AccGpuSyclIntel<TDim, TIdx>, TDim, TIdx, TKernelFnObj, TArgs...>
    {
    public:
        //-----------------------------------------------------------------------------
        template<typename TWorkDiv>
        ALPAKA_FN_HOST TaskKernelGpuSyclIntel(TWorkDiv && workDiv, TKernelFnObj const & kernelFnObj, TArgs const & ... args)
        : TaskKernelSyclImpl(std::forward<TWorkDiv>(workDiv), kernelFnObj, args...)
        {
            static_assert(
                Dim<typename std::decay<TWorkDiv>::type>::value == TDim::value,
                "The work division and the execution task have to be of the same dimensionality!");
        }

        //-----------------------------------------------------------------------------
        TaskKernelGpuSyclIntel(TaskKernelGpuSyclIntel const &) = default;
        //-----------------------------------------------------------------------------
        TaskKernelGpuSyclIntel(TaskKernelGpuSyclIntel &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGpuSyclIntel const &) -> TaskKernelGpuSyclIntel & = default;
        //-----------------------------------------------------------------------------
        auto operator=(TaskKernelGpuSyclIntel &&) -> TaskKernelGpuSyclIntel & = default;
        //-----------------------------------------------------------------------------
        ~TaskKernelGpuSyclIntel() = default;

    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL execution task accelerator type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct AccType<TaskKernelGpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = AccGpuSyclIntel<TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL execution task device type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct DevType<TaskKernelGpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = DevGpuSyclIntel;
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
        struct PltfType<TaskKernelGpuSyclIntel<TDim, TIdx, TKernelFnObj, TArgs...>>
        {
            using type = PltfGpuSyclIntel;
        };
    }
}

#endif
