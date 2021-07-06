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

#include <alpaka/core/Concepts.hpp>
#include <alpaka/math/min/Traits.hpp>

#include <sycl/sycl.hpp>

#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The SYCL library min.
        class MinGenericSycl : public concepts::Implements<ConceptMathMin, MinGenericSycl>
        {
        };

        namespace traits
        {
            //! The SYCL integral min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<MinGenericSycl, Tx, Ty, std::enable_if_t<std::is_integral_v<Tx> && std::is_integral_v<Ty>>>
            {
                auto operator()(MinGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return sycl::min(x, y);
                }
            };
            //! The SYCL mixed integral floating point min trait specialization.
            template<typename Tx, typename Ty>
            struct Min<MinGenericSycl, Tx, Ty, std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>
                                                                && !(std::is_integral_v<Tx> && std::is_integral_v<Ty>)>>
            {
                auto operator()(MinGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return sycl::fmin(x, y);
                }
            };
        }
    }
}

#endif
