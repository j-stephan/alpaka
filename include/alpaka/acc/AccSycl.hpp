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

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivSycl.hpp>
#include <alpaka/idx/gb/IdxGbSycl.hpp>
#include <alpaka/idx/bt/IdxBtSycl.hpp>
#include <alpaka/atomic/AtomicSycl.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathSycl.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynSycl.hpp>
//#include <alpaka/block/shared/st/BlockSharedMemStSycl.hpp>
#include <alpaka/block/sync/BlockSyncSycl.hpp>
//#include <alpaka/rand/RandSycl.hpp>
//#include <alpaka/time/TimeSycl.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/DevSycl.hpp>

#include <typeinfo>

namespace alpaka
{
    template<typename TDim, typename TIdx, typename TKernelFnObj, typename... TArgs>
    class TaskKernelSycl;

    //#############################################################################
    //! The SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL devices.
    template<
        typename TDim,
        typename TIdx>
    class AccSycl final :
        public WorkDivSycl<TDim, TIdx>,
        public gb::IdxGbSycl<TDim, TIdx>,
        public bt::IdxBtSycl<TDim, TIdx>,
        public AtomicHierarchy<AtomicSycl, AtomicSycl, AtomicSycl>,
        public math::MathSycl,
        public BlockSharedMemDynSycl,
        //public BlockSharedMemStSycl,
        public BlockSyncSycl<TDim>,
        //public rand::RandSycl,
        //public TimeSycl
        public concepts::Implements<ConceptAcc, AccSycl<TDim, TIdx>>
    {
    public:
        //-----------------------------------------------------------------------------
        AccSycl(
            Vec<TDim, TIdx> const & threadElemExtent,
            cl::sycl::nd_item<TDim::value> work_item,
            cl::sycl::accessor<unsigned char, 1,
                               cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> shared_acc,
            cl::sycl::accessor<int, 0,
                               cl::sycl::access::mode::atomic,
                               cl::sycl::access::target::local> pred_counter) :
                WorkDivSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbSycl<TDim, TIdx>{work_item},
                bt::IdxBtSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicSycl, AtomicSycl, AtomicSycl>{},
                math::MathSycl(),
                BlockSharedMemDynSycl{shared_acc},
                // BlockSharedMemStSycl(),
                BlockSyncSycl<TDim>{work_item, pred_counter}
                /*rand::RandSycl(),
                TimeSycl()*/
        {}

    public:
        //-----------------------------------------------------------------------------
        AccSycl(AccSycl const & rhs)
        : WorkDivSycl<TDim, TIdx>{rhs}
        , gb::IdxGbSycl<TDim, TIdx>{rhs}
        , bt::IdxBtSycl<TDim, TIdx>{rhs}
        , AtomicHierarchy<AtomicSycl, AtomicSycl, AtomicSycl>{rhs}
        , math::MathSycl{rhs}
        , BlockSharedMemDynSycl{rhs}
        , BlockSyncSycl<TDim>{rhs}
        {
        }
        //-----------------------------------------------------------------------------
        AccSycl(AccSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccSycl const &) -> AccSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccSycl &&) -> AccSycl & = delete;
        //-----------------------------------------------------------------------------
        ~AccSycl() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator type trait specialization.
        template<typename TDim, typename TIdx>
        struct AccType<AccSycl<TDim, TIdx>>
        {
            using type = AccSycl<TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL accelerator device properties get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccDevProps<AccSycl<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(DevSycl const & dev) -> AccDevProps<TDim, TIdx>
            {
                auto max_threads_dim =
                    dev.m_Device.get_info<cl::sycl::info::device::max_work_item_sizes>();
                return {
                    // m_multiProcessorCount
                    alpaka::core::clipCast<TIdx>(dev.m_Device.get_info<cl::sycl::info::device::max_compute_units>()),
                    // m_gridBlockExtentMax
                    extent::getExtentVecEnd<TDim>(
                        Vec<DimInt<3u>, TIdx>(
                            // FIXME: There is no SYCL way to determine these values
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
                    alpaka::core::clipCast<TIdx>(
                            dev.m_Device.get_info<cl::sycl::info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    dev.m_Device.get_info<cl::sycl::info::device::local_mem_size>()
                };
            }
        };

        //#############################################################################
        //! The SYCL accelerator name trait specialization.
        template<typename TDim, typename TIdx>
        struct GetAccName<AccSycl<TDim, TIdx>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccName() -> std::string
            {
                return "AccSycl<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
            }
        };

        //#############################################################################
        //! The SYCL accelerator device type trait specialization.
        template<typename TDim, typename TIdx>
        struct DevType<AccSycl<TDim, TIdx>>
        {
            using type = DevSycl;
        };

        //#############################################################################
        //! The SYCL accelerator dimension getter trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<AccSycl<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL accelerator execution task type trait specialization.
        template<typename TDim, typename TIdx, typename TWorkDiv, typename TKernelFnObj, typename... TArgs>
        struct CreateTaskKernel<AccSycl<TDim, TIdx>, TWorkDiv, TKernelFnObj, TArgs...>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto createTaskKernel(TWorkDiv const & workDiv, TKernelFnObj const & kernelFnObj,
                                                        TArgs const & ... args)
            {
                return TaskKernelSycl<TDim, TIdx, TKernelFnObj, TArgs...>(workDiv, kernelFnObj, args...);
            }
        };

        //#############################################################################
        //! The SYCL execution task platform type trait specialization.
        template<typename TDim, typename TIdx>
        struct PltfType<AccSycl<TDim, TIdx>>
        {
            using type = PltfSycl;
        };

        //#############################################################################
        //! The SYCL accelerator idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<AccSycl<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
