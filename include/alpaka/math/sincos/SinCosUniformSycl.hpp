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

#include <alpaka/math/sincos/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! sincos.
        class SinCosUniformSycl : public concepts::Implements<ConceptMathSinCos, SinCosUniformSycl>
        {
        public:
            using SinCosBase = SinCosUniformSycl;
        };

        namespace traits
        {
            //#############################################################################

            //! sincos trait specialization.
            template<typename TArg>
            struct SinCos<
                SinCosUniformSycl,
                TArg>
            {
                static auto sincos(
                    SinCosUniformSycl const &,
                    TArg const & arg,
                    TArg & result_sin,
                    TArg & result_cos)
                -> void
                {
                    result_sin = cl::sycl::sincos(arg, &result_cos);
                }
            };
        }
    }
}

#endif
