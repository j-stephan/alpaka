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

#include <alpaka/mem/access/Traits.hpp>
#include <alpaka/mem/buf/BufSycl.hpp>

namespace alpaka
{
    namespace mem
    {
        namespace access
        {
            //#############################################################################
            //! The SYCL buffer accessor.
            template<
                typename TElem,
                typename TDim,
                access::mode AccessMode,
                access::target AccessTarget>
            class AccessSycl
            {
            public:
                AccessSycl(cl::sycl::accessor<TElem, dim::Dim<TDim>::value,
                                              AccessMode, AccessTarget,
                                              cl::sycl::access::placeholder::true_t> access)
                : m_access{access}
                {}

            public:
                // using a placeholder here, binding it in the Kernel cgh later
                cl::sycl::accessor<TElem, dim::Dim<TDim>::value,
                                   static_cast<cl::sycl::access::mode>(AccessMode),
                                   static_cast<cl::sycl::access::target>(AccessTarget),
                                   cl::sycl::access::placeholder::true_t> m_access;
            };

            namespace traits
            {
                //#############################################################################
                //! The AccessSycl accessor get trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    access::mode AccessMode,
                    access::target AccessTarget>
                struct GetAccess<
                    mem::access::AccessSycl<TElem, TDim, AccessMode, AccessTarget>>
                {
                    ALPAKA_FN_HOST static auto getAccess(mem::buf::BufSycl<TElem, TDim, TIdx> & buf)
                    {
                        return mem::access::AccessSycl<TElem, TDim,
                                                       AccessMode, AccessTarget>
                            {
                                cl::sycl::accessor<TElem, dim::Dim<TDim>::value,
                                                   static_cast<cl::sycl::access::mode>(AccessMode),
                                                   static_cast<cl::sycl::access::target>(AccessTarget),
                                                   cl::sycl::access::placeholder::true_t>
                                                   {buf.m_buf};
                            };
                    }
                };
            }
        }
    }
}

#endif
