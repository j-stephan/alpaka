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

#include <alpaka/math/max/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library max.
        class MaxGenericSycl : public concepts::Implements<ConceptMathMax, MaxGenericSycl>
        {
        public:
            using MaxBase = MaxGenericSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxGenericSycl,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_integral_v<Tx>
                    && std::is_integral_v<Ty>>>
            {
                static auto max(
                    MaxGenericSycl const &,
                    Tx const & x,
                    Ty const & y)
                {
                    return cl::sycl::max(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point max trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Max<
                MaxGenericSycl,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_floating_point_v<Tx>
                    && std::is_floating_point_v<Ty>>>
            {
                static auto max(
                    MaxGenericSycl const &,
                    Tx const & x,
                    Ty const & y)
                {
                    return cl::sycl::fmax(x, y);
                }
            };
        }
    }
}

#endif
