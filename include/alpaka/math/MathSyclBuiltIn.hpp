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

#include <alpaka/math/abs/AbsSyclBuiltIn.hpp>
#include <alpaka/math/acos/AcosSyclBuiltIn.hpp>
#include <alpaka/math/asin/AsinSyclBuiltIn.hpp>
#include <alpaka/math/atan/AtanSyclBuiltIn.hpp>
#include <alpaka/math/atan2/Atan2SyclBuiltIn.hpp>
#include <alpaka/math/cbrt/CbrtSyclBuiltIn.hpp>
#include <alpaka/math/ceil/CeilSyclBuiltIn.hpp>
#include <alpaka/math/cos/CosSyclBuiltIn.hpp>
#include <alpaka/math/erf/ErfSyclBuiltIn.hpp>
#include <alpaka/math/exp/ExpSyclBuiltIn.hpp>
#include <alpaka/math/floor/FloorSyclBuiltIn.hpp>
#include <alpaka/math/fmod/FmodSyclBuiltIn.hpp>
#include <alpaka/math/log/LogSyclBuiltIn.hpp>
#include <alpaka/math/max/MaxSyclBuiltIn.hpp>
#include <alpaka/math/min/MinSyclBuiltIn.hpp>
#include <alpaka/math/pow/PowSyclBuiltIn.hpp>
#include <alpaka/math/remainder/RemainderSyclBuiltIn.hpp>
#include <alpaka/math/round/RoundSyclBuiltIn.hpp>
#include <alpaka/math/rsqrt/RsqrtSyclBuiltIn.hpp>
#include <alpaka/math/sin/SinSyclBuiltIn.hpp>
#include <alpaka/math/sqrt/SqrtSyclBuiltIn.hpp>
#include <alpaka/math/tan/TanSyclBuiltIn.hpp>
#include <alpaka/math/trunc/TruncSyclBuiltIn.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathSyclBuiltIn :
            public AbsSyclBuiltIn,
            public AcosSyclBuiltIn,
            public AsinSyclBuiltIn,
            public AtanSyclBuiltIn,
            public Atan2SyclBuiltIn,
            public CbrtSyclBuiltIn,
            public CeilSyclBuiltIn,
            public CosSyclBuiltIn,
            public ErfSyclBuiltIn,
            public ExpSyclBuiltIn,
            public FloorSyclBuiltIn,
            public FmodSyclBuiltIn,
            public LogSyclBuiltIn,
            public MaxSyclBuiltIn,
            public MinSyclBuiltIn,
            public PowSyclBuiltIn,
            public RemainderSyclBuiltIn,
            public RoundSyclBuiltIn,
            public RsqrtSyclBuiltIn,
            public SinSyclBuiltIn,
            public SqrtSyclBuiltIn,
            public TanSyclBuiltIn,
            public TruncSyclBuiltIn
        {};
    }
}

#endif
