/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <sycl/sycl.hpp>

#include <cstddef>

namespace alpaka
{
    //#############################################################################
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynGenericSycl : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynGenericSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynGenericSycl;

        //-----------------------------------------------------------------------------
        BlockSharedMemDynGenericSycl(sycl::accessor<std::byte, 1,
                                                        sycl::access::mode::read_write,
                                                        sycl::access::target::local> shared_acc)
        : acc{shared_acc}
        {}

        //-----------------------------------------------------------------------------
        BlockSharedMemDynGenericSycl(BlockSharedMemDynGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        BlockSharedMemDynGenericSycl(BlockSharedMemDynGenericSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynGenericSycl const &) -> BlockSharedMemDynGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynGenericSycl &&) -> BlockSharedMemDynGenericSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSharedMemDynGenericSycl() = default;

        sycl::accessor<std::byte, 1, sycl::access::mode::read_write, sycl::access::target::local> acc;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T>
        struct GetDynSharedMem<T, BlockSharedMemDynGenericSycl>
        {
            //-----------------------------------------------------------------------------
            static auto getMem(BlockSharedMemDynGenericSycl const & shared) -> T*
            {
                using namespace sycl;

                auto void_ptr = multi_ptr<void, access::address_space::local_space>{shared.acc};
                auto T_ptr = static_cast<multi_ptr<T, access::address_space::local_space>>(void_ptr);
                return T_ptr.get();
            }
        };
    }
}

#endif
