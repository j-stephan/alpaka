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
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/math/min/Traits.hpp>

#include <CL/sycl.hpp>

#include <type_traits>
#include <algorithm>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library min.
        class MinSycl
        {
        public:
            using MinBase = MinSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library integral min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinSycl,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_integral_v<Tx>
                    && std::is_integral_v<Ty>>>
            {
                static auto min(
                    MinSycl const & min,
                    Tx const & x,
                    Ty const & y)
                {
                    alpaka::ignore_unused(min);
                    return cl::sycl::min(x, y);
                }
            };
            //#############################################################################
            //! The standard library mixed integral floating point min trait specialization.
            template<
                typename Tx,
                typename Ty>
            struct Min<
                MinSycl,
                Tx,
                Ty,
                std::enable_if_t<
                    std::is_floating_point_v<Tx>
                    && std::is_floating_point_v<Ty>>>
            {
                static auto min(
                    MinSycl const & min,
                    Tx const & x,
                    Ty const & y)
                {
                    alpaka::ignore_unused(min);
                    return cl::sycl::fmin(x, y);
                }
            };
        }
    }
}

#endif
