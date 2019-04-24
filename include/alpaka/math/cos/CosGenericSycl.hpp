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
#    include <alpaka/math/cos/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL cos.
        class CosGenericSycl : public concepts::Implements<alpaka::math::ConceptMathCos, CosGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL cos trait specialization.
        template<typename TArg>
        struct Cos<experimental::math::CosGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto cos(experimental::math::CosGenericSycl const&, TArg const& arg)
            {
                return sycl::cos(arg);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
