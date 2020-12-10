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

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/math/sqrt/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sqrt.
        class SqrtGenericSycl : public concepts::Implements<ConceptMathSqrt, SqrtGenericSycl>
        {
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sqrt trait specialization.
            template<
                typename TArg>
            struct Sqrt<
                SqrtGenericSycl,
                TArg,
                std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                static auto sqrt(
                    SqrtGenericSycl const &,
                    TArg const & arg)
                {
                    return cl::sycl::sqrt(arg);
                }
            };
        }
    }
}

#endif
