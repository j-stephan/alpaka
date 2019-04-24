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

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/math/isinf/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL isfinite.
        class IsinfGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsinf, IsinfGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL isinf trait specialization.
        template<typename TArg>
        struct Isinf<experimental::math::IsinfGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            auto operator()(experimental::math::IsinfGenericSycl const&, TArg const& arg)
            {
                return sycl::isinf(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
