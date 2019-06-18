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

#include <alpaka/math/asin/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library asin.
        class AsinSyclBuiltIn
        {
        public:
            using AsinBase = AsinSyclBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library asin trait specialization.
            template<
                typename TArg>
            struct Asin<
                AsinSyclBuiltIn,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto asin(
                    AsinSyclBuiltIn const & asin,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(asin);
                    return cl::sycl::asin(arg);
                }
            };
        }
    }
}

#endif
