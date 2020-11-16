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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    class PltfGpuSyclIntelSycl;

    template<typename TElem, typename TDim, typename TIdx>
    class BufUniformSycl;

    //#############################################################################
    //! The SYCL device handle.
    class DevGpuSyclIntel : public DevUniformSycl
                           , public concepts::Implements<ConceptCurrentThreadWaitFor, DevGpuSyclIntel>
    {
        friend struct traits::GetDevByIdx<PltfGpuSyclIntelSycl>;
        friend struct traits::GetName<DevGpuSyclIntel>;
        friend struct traits::GetMemBytes<DevGpuSyclIntel>;

    protected:
        //-----------------------------------------------------------------------------
        DevGpuSyclIntel() = default;
    public:
        DevGpuSyclIntel(cl::sycl::device device, cl::sycl::context context, cl::sycl::queue queue)
        : DevUniformSycl(device, m_context, m_queue)
        {}

        //-----------------------------------------------------------------------------
        DevGpuSyclIntel(DevGpuSyclIntel const &) = default;
        //-----------------------------------------------------------------------------
        DevGpuSyclIntel(DevGpuSyclIntel &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevGpuSyclIntel const &) -> DevGpuSyclIntel & = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevGpuSyclIntel &&) -> DevGpuSyclIntel & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(DevGpuSyclIntel const & rhs) const -> bool
        {
            return DevUniformSycl::operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(DevGpuSyclIntel const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~DevGpuSyclIntel() = default;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevGpuSyclIntel, TElem, TDim, TIdx>
        {
            using type = BufUniformSycl<TElem, TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL device platform type trait specialization.
        template<>
        struct PltfType<DevGpuSyclIntel>
        {
            using type = PltfGpuSyclIntel;
        };
    }
}

#endif
