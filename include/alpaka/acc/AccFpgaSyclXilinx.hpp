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

#include <alpaka/atomic/AtomicGenericSycl.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp>
#include <alpaka/block/sync/BlockSyncGenericSycl.hpp>
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/idx/bt/IdxBtGenericSycl.hpp>
#include <alpaka/idx/gb/IdxGbGenericSycl.hpp>
#include <alpaka/intrinsic/IntrinsicGenericSycl.hpp>
#include <alpaka/kernel/TaskKernelFpgaSyclXilinx.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/math/MathGenericSycl.hpp>
#include <alpaka/pltf/PltfFpgaSyclXilinx.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/warp/WarpGenericSycl.hpp>
#include <alpaka/workdiv/WorkDivGenericSycl.hpp>
#include <alpaka/vec/Vec.hpp>

#include <sycl/sycl.hpp>

#include <string>

namespace alpaka
{
    //! The Xilinx FPGA SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL-capable Xilinx FPGAs.
    template<typename TDim, typename TIdx>
    class AccFpgaSyclXilinx :
        public WorkDivGenericSycl<TDim, TIdx>,
        public gb::IdxGbGenericSycl<TDim, TIdx>,
        public bt::IdxBtGenericSycl<TDim, TIdx>,
        public AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>,
        public math::MathGenericSycl,
        public BlockSharedMemDynGenericSycl,
        public BlockSyncGenericSycl<TDim>,
        public IntrinsicGenericSycl,
        public warp::WarpGenericSycl<TDim>,
        public concepts::Implements<ConceptAcc, AccFpgaSyclXilinx<TDim, TIdx>>
    {
    public:
#ifdef ALPAKA_SYCL_STREAM_ENABLED
        //-----------------------------------------------------------------------------
        AccFpgaSyclXilinx(
            Vec<TDim, TIdx> const & threadElemExtent,
            sycl::nd_item<TDim::value> work_item,
            sycl::accessor<std::byte, 1,
                               sycl::access::mode::read_write,
                               sycl::access::target::local> shared_acc,
            sycl::stream output_stream) :
                WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbGenericSycl<TDim, TIdx>{work_item},
                bt::IdxBtGenericSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{},
                math::MathGenericSycl{},
                BlockSharedMemDynGenericSycl{shared_acc},
                BlockSyncGenericSycl<TDim>{work_item}//,
                IntrinsicGenericSycl{},
                warp::WarpGenericSycl<TDim>{work_item},
                stream{output_stream}
        {}

        sycl::stream stream;
#else
        //-----------------------------------------------------------------------------
        AccFpgaSyclXilinx(
            Vec<TDim, TIdx> const & threadElemExtent,
            sycl::nd_item<TDim::value> work_item,
            sycl::accessor<std::byte, 1,
                               sycl::access::mode::read_write,
                               sycl::access::target::local> shared_acc) :
                WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbGenericSycl<TDim, TIdx>{work_item},
                bt::IdxBtGenericSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>{},
                math::MathGenericSycl{},
                BlockSharedMemDynGenericSycl{shared_acc},
                BlockSyncGenericSycl<TDim>{work_item},
                IntrinsicGenericSycl{},
                warp::WarpGenericSycl<TDim>{work_item}
        {}
#endif

        AccFpgaSyclXilinx(AccFpgaSyclXilinx const &) = delete;
        //-----------------------------------------------------------------------------
        AccFpgaSyclXilinx(AccFpgaSyclXilinx &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccFpgaSyclXilinx const &) -> AccFpgaSyclXilinx & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccFpgaSyclXilinx &&) -> AccFpgaSyclXilinx & = delete;
        //-----------------------------------------------------------------------------
        ~AccFpgaSyclXilinx() = default;
    };

    namespace traits
    {
        //! The SYCL accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = AccFpgaSyclXilinx<TDim, TIdx>;
        };

        //! The SYCL accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(DevFpgaSyclXilinx const& dev) -> AccDevProps<TDim, TIdx>
            {
                using namespace sycl;

                auto max_threads_dim = dev.m_device.get_info<info::device::max_work_item_sizes>();
                return {
                    // m_multiProcessorCount
                    alpaka::core::clipCast<TIdx>(dev.m_device.get_info<info::device::max_compute_units>()),
                    // m_gridBlockExtentMax
                    extent::getExtentVecEnd<TDim>(
                        Vec<DimInt<3u>, TIdx>(
                            // WARNING: There is no SYCL way to determine these values
                            std::numeric_limits<TIdx>::max(),
                            std::numeric_limits<TIdx>::max(),
                            std::numeric_limits<TIdx>::max())),
                    // m_gridBlockCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_blockThreadExtentMax
                    extent::getExtentVecEnd<TDim>(
                        Vec<DimInt<3u>, TIdx>(
                            alpaka::core::clipCast<TIdx>(max_threads_dim[2u]),
                            alpaka::core::clipCast<TIdx>(max_threads_dim[1u]),
                            alpaka::core::clipCast<TIdx>(max_threads_dim[0u]))),
                    // m_blockThreadCountMax
                    alpaka::core::clipCast<TIdx>(dev.m_device.get_info<info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    dev.m_device.get_info<info::device::local_mem_size>()
                };
            }
        };

        //! The Xilinx FPGA accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = TDim;
        };

        //! The Xilinx FPGA accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = TIdx;
        };

        //! The Xilinx FPGA accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccFpgaSyclXilinx<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //! The Xilinx FPGA accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = DevFpgaSyclXilinx;
        };

        //! The Xilinx FPGA accelerator execution task type trait specialization.
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

        //! The Xilinx FPGA execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccFpgaSyclXilinx<TDim, TIdx>>
        {
            using type = PltfFpgaSyclXilinx;
        };
    }
}

#endif
