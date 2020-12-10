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
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/math/remainder/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL remainder.
        class RemainderGenericSycl : public concepts::Implements<ConceptMathRemainder, RemainderGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL remainder trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Remainder<
                RemainderGenericSycl,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_floating_point_v<Tx>
                    && std::is_floating_point_v<Ty>>>
            {
                static auto remainder(
                    RemainderGenericSycl const &,
                    Tx const & x,
                    Ty const & y)
                {
                    return cl::sycl::remainder(x, y);
                }
            };
        }
    }
}

#endif
