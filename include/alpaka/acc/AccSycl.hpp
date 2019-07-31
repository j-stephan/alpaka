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
    namespace kernel
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelSycl;
    }
    namespace acc
    {
        //#############################################################################
        //! The SYCL accelerator.
        //!
        //! This accelerator allows parallel kernel execution on SYCL devices.
        template<
            typename TDim,
            typename TIdx>
        class AccSycl final :
            public workdiv::WorkDivSycl<TDim, TIdx>,
            public idx::gb::IdxGbSycl<TDim, TIdx>,
            public idx::bt::IdxBtSycl<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicSycl, // grid atomics
                atomic::AtomicSycl, // block atomics
                atomic::AtomicSycl  // thread atomics
            >,
            public math::MathSycl,
            public block::shared::dyn::BlockSharedMemDynSycl,
            //public block::shared::st::BlockSharedMemStSycl,
            public block::sync::BlockSyncSycl<TDim>
            //public rand::RandSycl,
            //public time::TimeSycl
        {
        public:
            //-----------------------------------------------------------------------------
            AccSycl(
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                cl::sycl::nd_item<TDim::value> work_item,
                cl::sycl::accessor<unsigned char, 1,
                                   cl::sycl::access::mode::read_write,
                                   cl::sycl::access::target::local> shared_acc,
                cl::sycl::accessor<int, 0,
                                   cl::sycl::access::mode::atomic,
                                   cl::sycl::access::target::local> pred_counter) :
                    workdiv::WorkDivSycl<TDim, TIdx>{threadElemExtent, work_item},
                    idx::gb::IdxGbSycl<TDim, TIdx>{work_item},
                    idx::bt::IdxBtSycl<TDim, TIdx>{work_item},
                    atomic::AtomicHierarchy<
                        atomic::AtomicSycl, // atomics between grids
                        atomic::AtomicSycl, // atomics between blocks
                        atomic::AtomicSycl  // atomics between threads
                    >{},
                    math::MathSycl(),
                    block::shared::dyn::BlockSharedMemDynSycl{shared_acc},
                    // block::shared::st::BlockSharedMemStSycl(),
                    block::sync::BlockSyncSycl<TDim>{work_item, pred_counter}
                    /*rand::RandSycl(),
                    time::TimeSycl()*/
            {}

        public:
            //-----------------------------------------------------------------------------
            AccSycl(AccSycl const & rhs)
            : workdiv::WorkDivSycl<TDim, TIdx>{rhs}
            , idx::gb::IdxGbSycl<TDim, TIdx>{rhs}
            , idx::bt::IdxBtSycl<TDim, TIdx>{rhs}
            , atomic::AtomicHierarchy<
                atomic::AtomicSycl,     // atomics between grids
                atomic::AtomicSycl,     // atomics between blocks
                atomic::AtomicSycl>{rhs}   // atomics between threads
            , math::MathSycl{rhs}
            , block::shared::dyn::BlockSharedMemDynSycl{rhs}
            , block::sync::BlockSyncSycl<TDim>{rhs}
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
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccSycl<TDim, TIdx>>
            {
                using type = acc::AccSycl<TDim, TIdx>;
            };
            //#############################################################################
            //! The SYCL accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccSycl<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevSycl const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    auto max_threads_dim =
                        dev.m_Device.get_info<cl::sycl::info::device::max_work_item_sizes>();
                    return {
                        // m_multiProcessorCount
                        alpaka::core::clipCast<TIdx>(
                                dev.m_Device.get_info<cl::sycl::info::device::max_compute_units>()),
                        // m_gridBlockExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                // FIXME: There is no SYCL way to determine these values
                                std::numeric_limits<TIdx>::max(),
                                std::numeric_limits<TIdx>::max(),
                                std::numeric_limits<TIdx>::max())),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        extent::getExtentVecEnd<TDim>(
                            vec::Vec<dim::DimInt<3u>, TIdx>(
                                alpaka::core::clipCast<TIdx>(max_threads_dim[2u]),
                                alpaka::core::clipCast<TIdx>(max_threads_dim[1u]),
                                alpaka::core::clipCast<TIdx>(max_threads_dim[0u]))),
                        // m_blockThreadCountMax
                        alpaka::core::clipCast<TIdx>(
                                dev.m_Device.get_info<cl::sycl::info::device::max_work_group_size>()),
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The SYCL accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccSycl<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccSycl<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccSycl<TDim, TIdx>>
            {
                using type = dev::DevSycl;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccSycl<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccSycl<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs const & ... args)
                {
                    return
                        kernel::TaskKernelSycl<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                args...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccSycl<TDim, TIdx>>
            {
                using type = pltf::PltfSycl;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccSycl<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
