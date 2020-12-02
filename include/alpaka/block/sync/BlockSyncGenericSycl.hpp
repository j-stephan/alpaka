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

#include <alpaka/block/sync/Traits.hpp>

#include <CL/sycl.hpp>

namespace alpaka
{
    //#############################################################################
    //! The SYCL block synchronization.
    template <typename TDim>
    class BlockSyncGenericSycl : public concepts::Implements<ConceptBlockSync, BlockSyncGenericSycl<TDim>>
    {
    public:
        using BlockSyncBase = BlockSyncGenericSycl<TDim>;

        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(cl::sycl::nd_item<TDim::value> work_item,
                             cl::sycl::ONEAPI::atomic_ref<int, cl::sycl::ONEAPI::memory_order::relaxed,
                                                          cl::sycl::ONEAPI::memory_scope::work_group,
                                                          cl::sycl::access::address_space::local_space> pred_counter)
        : my_item{work_item}
        , counter{pred_counter}
        {
        }
        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(BlockSyncGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        BlockSyncGenericSycl(BlockSyncGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncGenericSycl const &) -> BlockSyncGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncGenericSycl &&) -> BlockSyncGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncGenericSycl() = default;

        cl::sycl::nd_item<TDim::value> my_item;
        cl::sycl::ONEAPI::atomic_ref<int, cl::sycl::ONEAPI::memory_order::relaxed,
                                     cl::sycl::ONEAPI::memory_scope::work_group,
                                     cl::sycl::access::address_space::local_space> counter;
    };

    namespace traits
    {
        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreads<BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreads(BlockSyncGenericSycl<TDim> const & blockSync) -> void
            {
                blockSync.my_item.barrier();
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockCount, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                blockSync.my_item.barrier();
                
                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    blockSync.counter.store(0);
                blockSync.my_item.barrier(access::fence_space::local_space);

                if(predicate)
                    blockSync.counter.fetch_add(1);
                blockSync.my_item.barrier(access::fence_space::local_space);

                return blockSync.counter.load();
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                blockSync.my_item.barrier();

                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    blockSync.counter.store(1);
                blockSync.my_item.barrier(access::fence_space::local_space);
                
                if(!predicate)
                    blockSync.counter.fetch_and(0);
                blockSync.my_item.barrier(access::fence_space::local_space);

                return blockSync.counter.load();
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockOr, BlockSyncGenericSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const & blockSync, int predicate) -> int
            {
                using namespace cl::sycl;

                blockSync.my_item.barrier();

                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    blockSync.counter.store(0);
                blockSync.my_item.barrier(access::fence_space::local_space);

                if(predicate)
                    blockSync.counter.fetch_or(1);
                blockSync.my_item.barrier(access::fence_space::local_space);

                return blockSync.counter.load();
            }
        };
    }
}

#endif
