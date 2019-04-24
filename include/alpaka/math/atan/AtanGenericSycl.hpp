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
#    include <alpaka/math/atan/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL atan.
        class AtanGenericSycl : public concepts::Implements<alpaka::math::ConceptMathAtan, AtanGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL atan trait specialization.
        template<typename TArg>
        struct Atan<experimental::math::AtanGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto atan(experimental::math::AtanGenericSycl const&, TArg const& arg)
            {
                return sycl::atan(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
