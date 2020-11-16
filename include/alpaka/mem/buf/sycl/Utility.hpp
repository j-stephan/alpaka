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

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/extent/Traits.hpp>
#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! Returns a SYCL range with the correct dimensionality.
        template <int Dim, typename TExtent>
        auto get_sycl_range(TExtent const & extent)
        {
            using namespace cl::sycl;

            if constexpr(Dim == 1)
                return range<1>{extent::getWidth(extent)};
            else if constexpr(Dim == 2)
                return range<2>{extent::getWidth(extent), extent::getHeight(extent)};
            else
                return range<3>{extent::getWidth(extent), extent::getHeight(extent), extent::getDepth(extent)};
        }
    }
}

#endif
