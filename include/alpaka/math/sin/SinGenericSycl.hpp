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
#    include <alpaka/math/sin/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL sin.
        class SinGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSin, SinGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL sin trait specialization.
        template<typename TArg>
        struct Sin<experimental::math::SinGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto sin(experimental::math::SinGenericSycl const&, TArg const& arg)
            {
                return sycl::sin(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
