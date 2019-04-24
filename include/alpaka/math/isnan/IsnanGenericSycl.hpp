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
#    include <alpaka/math/isnan/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL isnan.
        class IsnanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathIsnan, IsnanGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL isnan trait specialization.
        template<typename TArg>
        struct Isnan<experimental::math::IsnanGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            auto operator()(experimental::math::IsnanGenericSycl const&, TArg const& arg)
            {
                return sycl::isnan(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
