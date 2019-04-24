/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Bert Wesarg, René Widera, Jan Stephan
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

#include <alpaka/math/trunc/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>


namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library trunc.
        class TruncSyclBuiltIn
        {
        public:
            using TruncBase = TruncSyclBuiltIn;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library trunc trait specialization.
            template<
                typename TArg>
            struct Trunc<
                TruncSyclBuiltIn,
                TArg,
                std::enable_if_t<std::is_floating_point_v<TArg>>>
            {
                static auto trunc(
                    TruncSyclBuiltIn const & trunc,
                    TArg const & arg)
                {
                    alpaka::ignore_unused(trunc);
                    return cl::sycl::trunc(arg);
                }
            };
        }
    }
}

#endif
