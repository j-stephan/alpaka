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

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <CL/sycl.hpp>

namespace alpaka
{
    //#############################################################################
    //! The SYCL block shared memory allocator.
    class BlockSharedMemDynUniformSycl : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynUniformSycl>
    {
    public:
        using BlockSharedMemDynBase = BlockSharedMemDynUniformSycl;

        //-----------------------------------------------------------------------------
        BlockSharedMemDynUniformSycl(cl::sycl::accessor<unsigned char, 1,
                                                        cl::sycl::access::mode::read_write,
                                                        cl::sycl::access::target::local> shared_acc)
        : acc{shared_acc}
        {}

        //-----------------------------------------------------------------------------
        BlockSharedMemDynUniformSycl(BlockSharedMemDynUniformSycl const &) = default;
        //-----------------------------------------------------------------------------
        BlockSharedMemDynUniformSycl(BlockSharedMemDynUniformSycl &&) = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynUniformSycl const &) -> BlockSharedMemDynUniformSycl & = delete;
        //-----------------------------------------------------------------------------
        auto operator=(BlockSharedMemDynUniformSycl &&) -> BlockSharedMemDynUniformSycl & = delete;
        //-----------------------------------------------------------------------------
        /*virtual*/ ~BlockSharedMemDynUniformSycl() = default;

        cl::sycl::accessor<unsigned char, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> acc;
    };

    namespace traits
    {
        //#############################################################################
        template<typename T>
        struct GetMem<T, BlockSharedMemDynUniformSycl>
        {
            //-----------------------------------------------------------------------------
            static auto getMem(BlockSharedMemDynUniformSycl const & shared) -> T *
            {
                auto ptr = static_cast<unsigned char*>(shared.acc.get_pointer());
                return reinterpret_cast<T*>(ptr);
            }
        };
    }
}

#endif
