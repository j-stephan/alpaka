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

#include <alpaka/math/abs/AbsSycl.hpp>
#include <alpaka/math/acos/AcosSycl.hpp>
#include <alpaka/math/asin/AsinSycl.hpp>
#include <alpaka/math/atan/AtanSycl.hpp>
#include <alpaka/math/atan2/Atan2Sycl.hpp>
#include <alpaka/math/cbrt/CbrtSycl.hpp>
#include <alpaka/math/ceil/CeilSycl.hpp>
#include <alpaka/math/cos/CosSycl.hpp>
#include <alpaka/math/erf/ErfSycl.hpp>
#include <alpaka/math/exp/ExpSycl.hpp>
#include <alpaka/math/floor/FloorSycl.hpp>
#include <alpaka/math/fmod/FmodSycl.hpp>
#include <alpaka/math/log/LogSycl.hpp>
#include <alpaka/math/max/MaxSycl.hpp>
#include <alpaka/math/min/MinSycl.hpp>
#include <alpaka/math/pow/PowSycl.hpp>
#include <alpaka/math/remainder/RemainderSycl.hpp>
#include <alpaka/math/round/RoundSycl.hpp>
#include <alpaka/math/rsqrt/RsqrtSycl.hpp>
#include <alpaka/math/sin/SinSycl.hpp>
#include <alpaka/math/sincos/SinCosSycl.hpp>
#include <alpaka/math/sqrt/SqrtSycl.hpp>
#include <alpaka/math/tan/TanSycl.hpp>
#include <alpaka/math/trunc/TruncSycl.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathSycl :
            public AbsSycl,
            public AcosSycl,
            public AsinSycl,
            public AtanSycl,
            public Atan2Sycl,
            public CbrtSycl,
            public CeilSycl,
            public CosSycl,
            public ErfSycl,
            public ExpSycl,
            public FloorSycl,
            public FmodSycl,
            public LogSycl,
            public MaxSycl,
            public MinSycl,
            public PowSycl,
            public RemainderSycl,
            public RoundSycl,
            public RsqrtSycl,
            public SinSycl,
            public SqrtSycl,
            public TanSycl,
            public TruncSycl
        {};
    }
}

#endif
