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
#    include <alpaka/math/round/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL round.
        class RoundGenericSycl : public concepts::Implements<alpaka::math::ConceptMathRound, RoundGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL round trait specialization.
        template<typename TArg>
        struct Round<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto round(experimental::math::RoundGenericSycl const&, TArg const& arg)
            {
                return sycl::round(arg);
            }
        };

        //! The SYCL lround trait specialization.
        template<typename TArg>
        struct Lround<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto lround(experimental::math::RoundGenericSycl const&, TArg const& arg)
            {
                return static_cast<long int>(sycl::round(arg));
            }
        };

        //! The SYCL llround trait specialization.
        template<typename TArg>
        struct Llround<experimental::math::RoundGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
        {
            static auto llround(experimental::math::RoundGenericSycl const&, TArg const& arg)
            {
                return static_cast<long long int>(sycl::round(arg));
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
