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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevUniformSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/buf/sycl/Utility.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/UniformSycl.hpp>

#include <CL/sycl.hpp>

#include <type_traits>
#include <vector>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The SYCL memory set trait.
        template<typename TElem>
        struct TaskSetSycl
        {
            //-----------------------------------------------------------------------------
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                cgh.depends_on(m_dependencies);
                cgh.memset(ptr, value, bytes);
            }

            TElem* const ptr;
            int value;
            std::size_t bytes;
            std::vector<cl::sycl::event> m_dependencies;
            std::shared_ptr<std::shared_mutex> mutex_ptr{std::make_shared<std::shared_mutex>()};
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TDev>
        struct CreateTaskMemset<TDim, TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView & view, std::uint8_t const& byte, TExtent const& ext)
            {
                using Type = Elem<TView>;
                constexpr auto TypeBytes = sizeof(Type);

                auto bytes = std::size_t{};
                if constexpr(Dim<TExtent>::value == 1)
                    bytes = extent::getWidth(ext) * TypeBytes;
                else if constexpr(Dim<TExtent>::value == 2)
                    bytes = extent::getWidth(ext) * extent::getHeight(ext) * TypeBytes;
                else
                    bytes = extent::getWidth(ext) * extent::getHeight(ext) * extent::getDepth(ext) * TypeBytes;

                return detail::TaskSetSycl<TDim, Type>{getPtrNative(view), static_cast<int>(byte), bytes, {}};
            }
        };
    }
}

#endif
