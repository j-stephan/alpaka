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

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <type_traits>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The SYCL block shared memory allocator.
                template <typename T>
                class BlockSharedMemDynSycl
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynSycl;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl() = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl(BlockSharedMemDynSycl const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl(BlockSharedMemDynSycl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynSycl const &) -> BlockSharedMemDynSycl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynSycl &&) -> BlockSharedMemDynSycl & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynSycl() = default;

                    cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
                                             cl::sycl::access::target::local> acc;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynSycl<T>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynSycl const & shared)
                        -> T *
                        {
                            auto ptr = shared.acc.get_pointer();
                            return static_cast<T*>(ptr);
                        }
                    };
                }
            }
        }
    }
}

#endif
