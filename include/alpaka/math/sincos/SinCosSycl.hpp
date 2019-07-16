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

#include <alpaka/math/sincos/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! sincos.
        class SinCosSycl
        {
        public:
            using SinCosBase = SinCosSycl;
        };

        namespace traits
        {
            //#############################################################################

            //! sincos trait specialization.
            template<typename TArg>
            struct SinCos<
                SinCosSycl,
                TArg>
            {
                static auto sincos(
                    SinCosSycl const & sincos_ctx,
                    TArg const & arg,
                    TArg & result_sin,
                    TArg & result_cos)
                -> void
                {
                    alpaka::ignore_unused(sincos_ctx);
                    result_sin = cl::sycl::sincos(arg, &result_cos);
                }
            };
        }
    }
}

#endif
