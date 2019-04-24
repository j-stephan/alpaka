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
#    include <alpaka/math/fmod/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <type_traits>

namespace alpaka
{
    namespace experimental::math
    {
        //! The SYCL fmod.
        class FmodGenericSycl : public concepts::Implements<alpaka::math::ConceptMathFmod, FmodGenericSycl>
        {
        };
    } // namespace experimental::math

    namespace math::traits
    {
        //! The SYCL fmod trait specialization.
        template<typename Tx, typename Ty>
        struct Fmod<
            experimental::math::FmodGenericSycl,
            Tx,
            Ty,
            std::enable_if_t<std::is_arithmetic_v<Tx> && std::is_arithmetic_v<Ty>>>
        {
            static auto fmod(experimental::math::FmodGenericSycl const&, Tx const& x, Ty const& y)
            {
                return sycl::fmod(x, y);
            }
        };
    } // namespace math::traits
} // namespace alpaka

#endif
