/* Copyright 2019 Jan Stephan
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
        class SqrtSycl
        {
        public:
            using SqrtBase = SqrtSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sqrt trait specialization.
            template<
                typename TArg>
            struct Sqrt<
                SqrtSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto sqrt(
                    SqrtSycl const & sqrt,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(sqrt);
                    return cl::sycl::sqrt(arg);
                }
            };
        }
    }
}

#endif
