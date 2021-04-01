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

#include <alpaka/core/Concepts.hpp>
#include <alpaka/math/remainder/Traits.hpp>

#include <sycl/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL remainder.
        class RemainderGenericSycl : public concepts::Implements<ConceptMathRemainder, RemainderGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL remainder trait specialization.
            template<typename Tx, typename Ty>
            struct Remainder<RemainderGenericSycl, Tx, Ty, std::enable_if_t<std::is_floating_point_v<Tx> && std::is_floating_point_v<Ty>>>
            {
                static auto remainder(RemainderGenericSycl const &, Tx const & x, Ty const & y)
                {
                    return sycl::remainder(x, y);
                }
            };
        }
    }
}

#endif
