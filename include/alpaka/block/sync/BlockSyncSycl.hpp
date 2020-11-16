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

namespace alpaka
{
    //#############################################################################
    //! The SYCL block synchronization.
    template <typename TDim>
    class BlockSyncSycl : public concepts::Implements<ConceptBlockSync, BlockSyncSycl<TDim>>
    {
    public:
        using BlockSyncBase = BlockSyncSycl<TDim>;

        //-----------------------------------------------------------------------------
        BlockSyncSycl(cl::sycl::nd_item<TDim::value> work_item,
                      cl::sycl::accessor<int, 0, cl::sycl::access::mode::atomic,
                                         cl::sycl::access::target::local> counter)
        : my_item{work_item}
        , pred_counter{counter}
        {
        }
        //-----------------------------------------------------------------------------
        BlockSyncSycl(BlockSyncSycl const &) = default;
        //-----------------------------------------------------------------------------
        BlockSyncSycl(BlockSyncSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncSycl const &) -> BlockSyncSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSyncSycl &&) -> BlockSyncSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSyncSycl() = default;

        cl::sycl::nd_item<TDim::value> my_item;
        cl::sycl::accessor<int, 0,
                           cl::sycl::access::mode::atomic,
                           cl::sycl::access::target::local> pred_counter;
    };

    namespace traits
    {
        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreads<BlockSyncSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreads(BlockSyncSycl<TDim> const & blockSync) -> void
            {
                blockSync.my_item.barrier();
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockCount, BlockSyncSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncSycl<TDim> const & blockSync, int predicate) -> int
            {
                // just copy the accessor, will refer to the same memory address
                auto counter = blockSync.pred_counter;
                blockSync.my_item.barrier();
                
                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    cl::sycl::atomic_store(counter, 0);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                if(predicate)
                    cl::sycl::atomic_fetch_add(counter, 1);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                return cl::sycl::atomic_load(counter);
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncSycl<TDim> const & blockSync, int predicate) -> int
            {
                // just copy the accessor, will refer to the same memory address
                auto counter = blockSync.pred_counter;
                blockSync.my_item.barrier();

                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    cl::sycl::atomic_store(counter, 1);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);
                
                if(!predicate)
                    cl::sycl::atomic_fetch_and(counter, 0);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                return cl::sycl::atomic_load(counter);
            }
        };

        //#############################################################################
        template<typename TDim>
        struct SyncBlockThreadsPredicate<BlockOr, BlockSyncSycl<TDim>>
        {
            //-----------------------------------------------------------------------------
            static auto syncBlockThreadsPredicate(BlockSyncSycl<TDim> const & blockSync, int predicate) -> int
            {
                // just copy the accessor, will refer to the same memory address
                auto counter = blockSync.pred_counter;
                blockSync.my_item.barrier();

                if(blockSync.my_item.get_local_linear_id(0) == 0)
                    cl::sycl::atomic_store(counter, 0);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                if(predicate)
                    cl::sycl::atomic_fetch_or(counter, 1);
                blockSync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                return cl::sycl::atomic_load(counter);
            }
        };
    }
}

#endif
