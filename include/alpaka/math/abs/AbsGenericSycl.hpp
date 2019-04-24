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
#    include <alpaka/math/abs/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL library abs.
        class AbsGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAbs, AbsGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL floating point abs trait specialization.
        template<typename TArg>
        struct Abs<experimental::math::AbsGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            static auto abs(experimental::math::AbsGenericSycl const&, TArg const& arg)
            {
                return sycl::fabs(arg);
            }
        };

        //! The SYCL integral abs trait specialization.
        template<typename TArg>
        struct Abs<experimental::math::AbsGenericSycl, TArg, std::enable_if_t<std::is_integral_v<TArg>>>
        {
            static auto abs(experimental::math::AbsGenericSycl const&, TArg const& arg)
            {
                return sycl::abs(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
