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
#include <alpaka/math/sqrt/Traits.hpp>

#include <sycl/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The SYCL library sqrt.
        class SqrtGenericSycl : public concepts::Implements<ConceptMathSqrt, SqrtGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL sqrt trait specialization.
            template<typename TArg>
            struct Sqrt<SqrtGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                static auto sqrt(SqrtGenericSycl const &, TArg const & arg)
                {
                    return sycl::sqrt(arg);
                }
            };
        }
    }
}

#endif
