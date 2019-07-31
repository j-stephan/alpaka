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
                class BlockSharedMemDynSycl
                {
                public:
                    using BlockSharedMemDynBase = BlockSharedMemDynSycl;

                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl(cl::sycl::accessor<unsigned char, 1,
                                            cl::sycl::access::mode::read_write,
                                            cl::sycl::access::target::local>
                                            shared_acc)
                    : acc{shared_acc}
                    {}

                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl(BlockSharedMemDynSycl const &) = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynSycl(BlockSharedMemDynSycl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynSycl const &) -> BlockSharedMemDynSycl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynSycl &&) -> BlockSharedMemDynSycl & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynSycl() = default;

                    cl::sycl::accessor<unsigned char, 1,
                                       cl::sycl::access::mode::read_write,
                                       cl::sycl::access::target::local> acc;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynSycl>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynSycl const & shared)
                        -> T *
                        {
                            auto ptr = static_cast<unsigned char*>(shared.acc.get_pointer());
                            return reinterpret_cast<T*>(ptr);
                        }
                    };
                }
            }
        }
    }
}

#endif
