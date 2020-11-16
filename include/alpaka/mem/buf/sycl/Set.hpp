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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/buf/sycl/Utility.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    class DevSycl;

    namespace detail
    {
        //#############################################################################
        //! The SYCL memory set trait.
        template<typename TDim>
        struct TaskSetSycl
        {
            //-----------------------------------------------------------------------------
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                auto acc = buf.template get_access<cl::sycl::access::mode::write>(cgh, range);
                cgh.fill(acc, byte);
            }

            cl::sycl::buffer<std::uint8_t, TDim::value> buf;
            std::uint8_t byte;
            cl::sycl::range<TDim::value> range;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL device memory set trait specialization.
        template<typename TDim>
        struct CreateTaskMemset<TDim, DevSycl>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView & view, std::uint8_t const & byte, TExtent const & extent)
            {
                using elem_type = Elem<TView>;
                constexpr auto elem_size = sizeof(elem_type);

                // multiply original range with sizeof(elem_type)
                const auto old_buf_range = view.m_buf.get_range();
                const auto new_buf_range = old_buf_range * elem_size;

                const auto sycl_range = detail::get_sycl_range<Dim<TExtent>::value>(extent);
                const auto byte_range = sycl_range * elem_size;

                // reinterpret the original buffer as a byte buffer
                auto new_buf = view.m_buf.template reinterpret<std::uint8_t>(new_buf_range);

                return detail::TaskSetSycl<TDim>{new_buf, byte, byte_range};
            }
        };
    }
}

#endif
