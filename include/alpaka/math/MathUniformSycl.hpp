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

#if !BOOST_LANG_SYCL
    #error If ALPAKA_ACC_SYCL_ENABLED is set, the compiler has to support SYCL!
#endif

#include <alpaka/math/abs/AbsUniformSycl.hpp>
#include <alpaka/math/acos/AcosUniformSycl.hpp>
#include <alpaka/math/asin/AsinUniformSycl.hpp>
#include <alpaka/math/atan/AtanUniformSycl.hpp>
#include <alpaka/math/atan2/Atan2UniformSycl.hpp>
#include <alpaka/math/cbrt/CbrtUniformSycl.hpp>
#include <alpaka/math/ceil/CeilUniformSycl.hpp>
#include <alpaka/math/cos/CosUniformSycl.hpp>
#include <alpaka/math/erf/ErfUniformSycl.hpp>
#include <alpaka/math/exp/ExpUniformSycl.hpp>
#include <alpaka/math/floor/FloorUniformSycl.hpp>
#include <alpaka/math/fmod/FmodUniformSycl.hpp>
#include <alpaka/math/log/LogUniformSycl.hpp>
#include <alpaka/math/max/MaxUniformSycl.hpp>
#include <alpaka/math/min/MinUniformSycl.hpp>
#include <alpaka/math/pow/PowUniformSycl.hpp>
#include <alpaka/math/remainder/RemainderUniformSycl.hpp>
#include <alpaka/math/round/RoundUniformSycl.hpp>
#include <alpaka/math/rsqrt/RsqrtUniformSycl.hpp>
#include <alpaka/math/sin/SinUniformSycl.hpp>
#include <alpaka/math/sincos/SinCosUniformSycl.hpp>
#include <alpaka/math/sqrt/SqrtUniformSycl.hpp>
#include <alpaka/math/tan/TanUniformSycl.hpp>
#include <alpaka/math/trunc/TruncUniformSycl.hpp>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The mathematical operation specifics.
    namespace math
    {
        //#############################################################################
        //! The standard library math trait specializations.
        class MathUniformSycl :
            public AbsUniformSycl,
            public AcosUniformSycl,
            public AsinUniformSycl,
            public AtanUniformSycl,
            public Atan2UniformSycl,
            public CbrtUniformSycl,
            public CeilUniformSycl,
            public CosUniformSycl,
            public ErfUniformSycl,
            public ExpUniformSycl,
            public FloorUniformSycl,
            public FmodUniformSycl,
            public LogUniformSycl,
            public MaxUniformSycl,
            public MinUniformSycl,
            public PowUniformSycl,
            public RemainderUniformSycl,
            public RoundUniformSycl,
            public RsqrtUniformSycl,
            public SinUniformSycl,
            public SqrtUniformSycl,
            public TanUniformSycl,
            public TruncUniformSycl
        {};
    }
}

#endif
