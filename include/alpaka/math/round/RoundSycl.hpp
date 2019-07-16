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

#include <alpaka/math/round/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library round.
        class RoundSycl
        {
        public:
            using RoundBase = RoundSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Round<
                RoundSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto round(
                    RoundSycl const & round,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(round);
                    return cl::sycl::round(arg);
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Lround<
                RoundSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto lround(
                    RoundSycl const & lround,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(lround);
                    return static_cast<long int>(cl::sycl::round(arg));
                }
            };
            //#############################################################################
            //! The standard library round trait specialization.
            template<
                typename TArg>
            struct Llround<
                RoundSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto llround(
                    RoundSycl const & llround,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(llround);
                    return static_cast<long long int>(cl::sycl::round(arg));
                }
            };
        }
    }
}

#endif
