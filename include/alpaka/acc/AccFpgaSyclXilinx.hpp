/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#include <alpaka/acc/AccGenericSycl.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/TaskKernelFpgaSyclXilinx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/PltfFpgaSyclXilinx.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <sycl/sycl.hpp>

#include <string>

namespace alpaka
{
    template <typename TDim, typename TIdx>
    class AccFpgaSyclXilinx : public AccGenericSycl<TDim, TIdx>
                            , public concepts::Implements<ConceptAcc, AccFpgaSyclXilinx<TDim, TIdx>>
    {
    public:
#ifdef ALPAKA_SYCL_STREAM_ENABLED
        AccFpgaSyclXilinx(Vec<TDim, TIdx> const & threadElemExtent, sycl::nd_item<TDim::value> work_item,
                          sycl::accessor<std::byte, 1, sycl::access::mode::read_write,
                                             sycl::access::target::local> shared_acc,
                          sycl::stream output_stream)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc, output_stream)
        {}
#else
        AccFpgaSyclXilinx(Vec<TDim, TIdx> const & threadElemExtent, sycl::nd_item<TDim::value> work_item,
                          sycl::accessor<std::byte, 1, sycl::access::mode::read_write,
                                             sycl::access::target::local> shared_acc)
        : AccGenericSycl<TDim, TIdx>(threadElemExtent, work_item, shared_acc)
        {}
#endif

        AccFpgaSyclXilinx(AccFpgaSyclXilinx const&) = delete;        
        auto operator=(AccFpgaSyclXilinx const&) -> AccFpgaSyclXilinx& = delete;

        AccFpgaSyclXilinx(AccFpgaSyclXilinx&&) = delete;
        auto operator=(AccFpgaSyclXilinx&&) -> AccFpgaSyclXilinx& = delete;

        ~AccFpgaSyclXilinx() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccFpgaSyclXilinx<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The SYCL accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = DevFpgaSyclXilinx;
        };

        //#############################################################################
        //! The SYCL accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccFpgaSyclXilinx<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(TWorkDiv const & workDiv, TKernelFnObj const & kernelFnObj,
                                                        TArgs const & ... args)
            {
                return TaskKernelFpgaSyclXilinx<TDim, TIdx, TKernelFnObj, TArgs...>{workDiv, kernelFnObj, args...};
            }
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = PltfFpgaSyclXilinx;
        };
    }
}

#endif
