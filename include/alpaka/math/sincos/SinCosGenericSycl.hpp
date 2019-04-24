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
#    include <alpaka/math/sincos/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL sincos.
        class SinCosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathSinCos, SinCosGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL sincos trait specialization.
        template<typename TArg>
        struct SinCos<experimental::math::SinCosGenericSycl, TArg, std::enable_if_t<std::is_floating_point_v<TArg>>>
        {
            static auto sincos(
                experimental::math::SinCosGenericSycl const&,
                TArg const& arg,
                TArg& result_sin,
                TArg& result_cos) -> void
            {
                result_sin = sycl::sincos(arg, &result_cos);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
