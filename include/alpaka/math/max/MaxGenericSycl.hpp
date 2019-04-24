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

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/math/max/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL max.
        class MaxGenericSycl : public concepts::Implements<alpaka::math::ConceptMathMax, MaxGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL integral max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<
            experimental::math::MaxGenericSycl,
            Tx,
            Ty,
            std::enable_if_t<std::is_integral_v<Tx> && std::is_integral_v<Ty>>>
        {
            static auto max(experimental::math::MaxGenericSycl const&, Tx const& x, Ty const& y)
            {
                return sycl::max(x, y);
            }
        };

        //! The SYCL mixed integral floating point max trait specialization.
        template<typename Tx, typename Ty>
        struct Max<
            experimental::math::MaxGenericSycl,
            Tx,
            Ty,
            std::enable_if_t<
                std::is_arithmetic_v<
                    Tx> && std::is_arithmetic_v<Ty> && !(std::is_integral_v<Tx> && std::is_integral_v<Ty>)>>
        {
            static auto max(experimental::math::MaxGenericSycl const&, Tx const& x, Ty const& y)
            {
                return sycl::fmax(x, y);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
