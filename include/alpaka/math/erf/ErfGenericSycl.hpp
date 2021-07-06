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
#include <alpaka/math/erf/Traits.hpp>

#include <sycl/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //! The SYCL erf.
        class ErfGenericSycl : public concepts::Implements<ConceptMathErf, ErfGenericSycl>
        {
        };

        namespace traits
        {
            //! The SYCL erf trait specialization.
            template<typename TArg>
            struct Erf<ErfGenericSycl, TArg, std::enable_if_t<std::is_arithmetic_v<TArg>>>
            {
                auto operator()(ErfGenericSycl const &, TArg const & arg)
                {
                    return sycl::erf(arg);
                }
            };
        }
    }
}

#endif
