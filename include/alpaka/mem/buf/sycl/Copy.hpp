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

#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/buf/sycl/Utility.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/UniformSycl.hpp>

#include <CL/sycl.hpp>

#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The SYCL memory copy trait.
        template <typename TElem>
        struct TaskCopySycl
        {
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                cgh.depends_on(m_dependencies);
                cgh.memcpy(dst_ptr, src_ptr, bytes);
            }

            TElem const* const src_ptr;
            TElem* const dst_ptr;
            std::size_t bytes;
            std::vector<cl::sycl::event> m_dependencies;
            std::shared_ptr<std::shared_mutex> mutex_ptr{std::make_shared<std::shared_mutex>()};
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        //#############################################################################
        //! The SYCL memory copy trait specialization.
        template<typename TDim, typename TDevDst, typename TDevSrc>
        struct CreateTaskMemcpy<TDim, TDevDst, TDevSrc,
                                // at least one of the partners has to be a SYCL device
                                std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDevDst> ||
                                                 std::is_base_of_v<DevUniformSycl, TDevSrc>>>
        {
            static_assert((std::is_base_of_v<DevUniformSycl, TDevDst> || std::is_same_v<DevCpu, TDevDst>) &&
                          (std::is_base_of_v<DevUniformSycl, TDevSrc> || std::is_same_v<DevCpu, TDevSrc>),
                          "Invalid device selected for copying. Only SYCL devices and CPU devices are supported.");
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc & viewSrc, TExtent const & ext)
            {
                using SrcType = Elem<std::remove_const_t<TViewSrc>>;
                constexpr auto SrcBytes = sizeof(SrcType);

                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, SrcType>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                auto bytes = std::size_t{};

                if constexpr(Dim<TExtent>::value == 1)
                    bytes = extent::getWidth(ext) * SrcBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = extent::getWidth(ext) * extent::getHeight(ext) * SrcBytes;
                else
                    bytes = extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext) * SrcBytes;

                return detail::TaskCopySycl<SrcType>{getPtrNative(viewSrc), getPtrNative(viewDst), bytes, {}};
            }
        };
    }
}

#endif
