/* Copyright 2020 Jan Stephan
 * 
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_ONEAPI)

#include <alpaka/core/Common.hpp>

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/dev/DevFpgaSyclIntel.hpp>
#include <alpaka/vec/Vec.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/mem/buf/Traits.hpp>

#include <CL/sycl.hpp>

#include <memory>
#include <type_traits>

namespace alpaka
{
    //#############################################################################
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    class BufFpgaSyclIntel : public BufUniformSycl<TElem, TDim, TIdx>
    {
        friend struct traits::GetDev<BufFpgaSyclIntel<TElem, TDim, TIdx>>;

        template <typename TIdxIntegralConst>
        friend struct extent::traits::GetExtent<TIndexIntegralConst, BufFpgaSyclIntel<TElem, TDim, TIdx>>;

        friend struct GetPitchBytes<DimInt<TDim::value - 1u>, BufFpgaSyclIntel<TElem, TDim, TIdx>>;

        static_assert(std::is_const<TElem>::value,
                      "The elem type of the buffer can not be const because the C++ Standard forbids containers of const elements!");

        static_assert(!std::is_const<TIdx>::value, "The idx type of the buffer can not be const!");

    public:
        //-----------------------------------------------------------------------------
        //! Constructor
        template<typename TExtent>
        ALPAKA_FN_HOST BufFpgaSyclIntel(DevFpgaSyclIntel const & dev, TElem* ptr, TIdx const& pitchBytes,
                                      TExtent const& extent)
        : BufUniformSycl(dev, ptr, pitchBytes, extent)
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

            static_assert(TDim::value == Dim<TExtent>::value,
                          "The dimensionality of TExtent and the dimensionality of the TDim template parameter have to be identical!");

            static_assert(std::is_same<TIdx, Idx<TExtent>>::value,
                          "The idx type of TExtent and the TIdx template parameter have to be identical!");
        }

        ALPAKA_FN_HOST ~BufFpgaSyclIntel() = default;
        ALPAKA_FN_HOST BufFpgaSyclIntel(BufFpgaSyclIntel const&) = default;
        ALPAKA_FN_HOST auto operator=(BufFpgaSyclIntel const&) -> BufFpgaSyclIntel& = default;
        ALPAKA_FN_HOST BufFpgaSyclIntel(BufFpgaSyclIntel&&) = default;
        ALPAKA_FN_HOST auto operator=(BufFpgaSyclIntel&&) -> BufFpgaSyclIntel = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The BufFpgaSyclIntel device type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct DevType<BufFpgaSyclIntel<TElem, TDim, TIdx>>
        {
            using type = DevFpgaSyclIntel;
        };
    }
}

#endif
