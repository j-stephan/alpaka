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

#include <alpaka/math/abs/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library abs.
        class AbsSycl
        {
        public:
            using AbsBase = AbsSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The SYCL built in abs trait specialization.
            template<
                typename TArg>
            struct Abs<
                AbsSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto abs(
                    AbsSycl const & abs,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(abs);
                    return cl::sycl::fabs(arg);
                }
            };
        }
    }
}

#endif