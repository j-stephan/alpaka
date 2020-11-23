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

#include <alpaka/math/sin/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library sin.
        class SinUniformSycl : public concepts::Implements<ConceptMathSin, SinUniformSycl>
        {
        public:
            using SinBase = SinUniformSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library sin trait specialization.
            template<
                typename TArg>
            struct Sin<
                SinUniformSycl,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto sin(
                    SinUniformSycl const &,
                    TArg const & arg)
                {
                    return cl::sycl::sin(arg);
                }
            };
        }
    }
}

#endif
