/* Copyright 2019 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

// FIXME: None of the following actually obeys const-correctness imposed by
// Alpaka's API. Unfortunately we need const-casts everywhere.

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/block/sync/Traits.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The SYCL block synchronization.
            template <typename TDim>
            class BlockSyncSycl
            {
            public:
                using BlockSyncBase = BlockSyncSycl<TDim>;

                //-----------------------------------------------------------------------------
                BlockSyncSycl(cl::sycl::nd_item<TDim::value> work_item,
                              cl::sycl::accessor<int, 0,
                                                 cl::sycl::access::mode::atomic,
                                                 cl::sycl::access::target::local> counter)
                : my_item{work_item}
                , pred_counter{counter}
                {
                }
                //-----------------------------------------------------------------------------
                BlockSyncSycl(BlockSyncSycl const &) = delete;
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
                struct SyncBlockThreads<
                    BlockSyncSycl<TDim>>
                {
                    //-----------------------------------------------------------------------------
                    static auto syncBlockThreads(
                        block::sync::BlockSyncSycl<TDim> const & blockSync)
                    -> void
                    {
                        // since barrier() is const we don't need const_cast here
                        blockSync.my_item.barrier();
                    }
                };

                //#############################################################################
                template<typename TDim>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::Count,
                    BlockSyncSycl<TDim>>
                {
                    //-----------------------------------------------------------------------------
                    static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncSycl<TDim> const & blockSync,
                        int predicate)
                    -> int
                    {
                        auto block_sync = *(const_cast<block::sync::BlockSyncSycl<TDim>*>(&blockSync));
                        block_sync.my_item.barrier();

                        if(block_sync.my_item.get_local_linear_id(0) == 0)
                            cl::sycl::atomic_store(block_sync.pred_counter, 0);

                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                        if(predicate)
                            cl::sycl::atomic_fetch_add(block_sync.pred_counter, 1);

                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                        return cl::sycl::atomic_load(block_sync.pred_counter);
                    }
                };

                //#############################################################################
                template<typename TDim>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalAnd,
                    BlockSyncSycl<TDim>>
                {
                    //-----------------------------------------------------------------------------
                    static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncSycl<TDim> const & blockSync,
                        int predicate)
                    -> int
                    {
                        auto block_sync = *(const_cast<block::sync::BlockSyncSycl<TDim>*>(&blockSync));
                        block_sync.my_item.barrier();

                        if(block_sync.my_item.get_local_linear_id(0) == 0)
                            cl::sycl::atomic_store(block_sync.pred_counter, 1);
                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);
                        
                        if(!predicate)
                            cl::sycl::atomic_fetch_and(block_sync.pred_counter, 0);
                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                        return cl::sycl::atomic_load(block_sync.pred_counter);
                    }
                };

                //#############################################################################
                template<typename TDim>
                struct SyncBlockThreadsPredicate<
                    block::sync::op::LogicalOr,
                    BlockSyncSycl<TDim>>
                {
                    //-----------------------------------------------------------------------------
                    static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncSycl<TDim> const & blockSync,
                        int predicate)
                    -> int
                    {
                        auto block_sync = *(const_cast<block::sync::BlockSyncSycl<TDim>*>(&blockSync));
                        block_sync.my_item.barrier();

                        if(block_sync.my_item.get_local_linear_id(0) == 0)
                            cl::sycl::atomic_store(block_sync.pred_counter, 0);
                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                        if(predicate)
                            cl::sycl::atomic_fetch_or(block_sync.pred_counter, 1);
                        block_sync.my_item.barrier(cl::sycl::access::fence_space::local_space);

                        return cl::sycl::atomic_load(block_sync.pred_counter);
                    }
                };
            }
        }
    }
}

#endif
