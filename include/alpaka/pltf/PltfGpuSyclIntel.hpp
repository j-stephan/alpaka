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

#include <alpaka/core/Sycl.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevGpuSyclIntel.hpp>
#include <alpaka/pltf/PltfUniformSycl.hpp>

#include <CL/sycl.hpp>

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace alpaka
{
    namespace detail
    {
        struct intel_gpu_selector : cl::sycl::device_selector
        {
            auto operator()(const cl::sycl::device& dev) const -> int override
            {
                const auto vendor = dev.get_info<cl::sycl::info::device::vendor>();
                const auto is_intel_gpu = (vendor.find("HD Graphics Neo") != std::string::npos) && dev.is_gpu();

                return is_intel_gpu ? 1 : -1;
            }
        };
    }

    //#############################################################################
    //! The SYCL device manager.
    class PltfGpuSyclIntel : public PltfUniformSycl
                           , public concepts::Implements<ConceptPltf, PltfGpuSyclIntel>
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST PltfGpuSyclIntel() = delete;

        using selector = detail::intel_gpu_selector;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device manager device type trait specialization.
        template<>
        struct DevType<PltfGpuSyclIntel>
        {
            using type = DevGpuSyclIntel;
        };
    }
}

#endif
