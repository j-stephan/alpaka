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

#include <alpaka/math/atan2/Traits.hpp>

#include <CL/sycl.hpp>
#include <type_traits>

namespace alpaka
{
    namespace math
    {
        //#############################################################################
        //! The standard library atan2.
        class Atan2GenericSycl : concepts::Implements<ConceptMathAtan2, Atan2GenericSycl>
        {
        public:
            using Atan2Base = Atan2GenericSycl;
        };

        namespace traits
        {
            //#############################################################################
            //! The standard library atan2 trait specialization.
            template<
                typename Ty,
                typename Tx>
            struct Atan2<
                Atan2GenericSycl,
                Ty,
                Tx,
                std::enable_if_t<
                    std::is_floating_point_v<Ty>
                    && std::is_floating_point_v<Tx>>>
            {
                static auto atan2(
                    Atan2GenericSycl const &,
                    Ty const & y,
                    Tx const & x)
                {
                    return cl::sycl::atan2(y, x);
                }
            };
        }
    }
}

#endif
