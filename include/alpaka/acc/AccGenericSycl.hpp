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

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

// Base classes.
#include <alpaka/workdiv/WorkDivSycl.hpp>
#include <alpaka/idx/gb/IdxGbSycl.hpp>
#include <alpaka/idx/bt/IdxBtSycl.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/atomic/AtomicUniformSycl.hpp>
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

#include <string>
#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL devices.
    template<typename TDim, typename TIdx>
    class AccGenericSycl :
        public WorkDivUniformSycl<TDim, TIdx>,
        public gb::IdxGbUniformSycl<TDim, TIdx>,
        public bt::IdxBtUniformSycl<TDim, TIdx>,
        public AtomicHierarchy<AtomicUniformSycl, AtomicUniformSycl, AtomicUniformSycl>,
        public math::MathUniformSycl,
        public BlockSharedMemDynUniformSycl,
        //public BlockSharedMemStUniformSycl,
        public BlockSyncUniformSycl<TDim>,
        //public rand::RandUniformSycl,
        //public TimeUniformSycl
        public concepts::Implements<ConceptAcc, AccGenericSycl<TDim, TIdx>>
    {
    public:
        //-----------------------------------------------------------------------------
        AccGenericSycl(
            Vec<TDim, TIdx> const & threadElemExtent,
            cl::sycl::nd_item<TDim::value> work_item,
            cl::sycl::accessor<unsigned char, 1,
                               cl::sycl::access::mode::read_write,
                               cl::sycl::access::target::local> shared_acc,
            cl::sycl::accessor<int, 0,
                               cl::sycl::access::mode::atomic,
                               cl::sycl::access::target::local> pred_counter) :
                WorkDivUniformSycl<TDim, TIdx>{threadElemExtent, work_item},
                gb::IdxGbUniformSycl<TDim, TIdx>{work_item},
                bt::IdxBtUniformSycl<TDim, TIdx>{work_item},
                AtomicHierarchy<AtomicUniformSycl, AtomicUniformSycl, AtomicUniformSycl>{},
                math::MathUniformSycl(),
                BlockSharedMemDynUniformSycl{shared_acc},
                // BlockSharedMemStUniformSycl(),
                BlockSyncUniformSycl<TDim>{work_item, pred_counter}
                /*rand::RandUniformSycl(),
                TimeUniformSycl()*/
        {}

        //-----------------------------------------------------------------------------
        AccGenericSycl(AccGenericSycl const & rhs)
        : WorkDivUniformSycl<TDim, TIdx>{rhs}
        , gb::IdxGbUniformSycl<TDim, TIdx>{rhs}
        , bt::IdxBtUniformSycl<TDim, TIdx>{rhs}
        , AtomicHierarchy<AtomicUniformSycl, AtomicUniformSycl, AtomicUniformSycl>{rhs}
        , math::MathUniformSycl{rhs}
        , BlockSharedMemDynUniformSycl{rhs}
        , BlockSyncUniformSycl<TDim>{rhs}
        {
        }
        //-----------------------------------------------------------------------------
        AccGenericSycl(AccGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccGenericSycl const &) -> AccGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(AccGenericSycl &&) -> AccGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        ~AccGenericSycl() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx>
        struct AccType<TAcc, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc>>>
        {
            using type = TAcc;
        };

        //#############################################################################
        //! The SYCL accelerator device properties get trait specialization.
        template<typename TAcc, typename TDim, typename TIdx>
        struct GetAccDevProps<TAcc, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getAccDevProps(typename DevType<TAcc>::type const & dev) -> AccDevProps<TDim, TIdx>
            {
                using namespace cl::sycl;

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
                    alpaka::core::clipCast<TIdx>(
                            dev.m_Device.get_info<info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    dev.m_dev.get_info<info::device::local_mem_size>()
                };
            }
        };

        //#############################################################################
        //! The SYCL accelerator dimension getter trait specialization.
        template<typename TAcc, typename TDim, typename TIdx>
        struct DimType<TAcc, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc>>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL accelerator idx type trait specialization.
        template<typename TAcc, typename TDim, typename TIdx>
        struct IdxType<TAcc, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc>>>
        {
            using type = TIdx;
        };
    }
}

#endif
