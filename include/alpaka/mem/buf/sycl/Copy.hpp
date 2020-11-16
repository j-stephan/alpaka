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

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/buf/sycl/Utility.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>

#include <set>
#include <tuple>

namespace alpaka
{
    namespace detail
    {
        //#############################################################################
        //! The SYCL memory copy type.
        enum class sycl_copy_type
        {
            h2d,
            d2h,
            d2d
        };

        //#############################################################################
        //! The SYCL memory copy trait.
        template<typename TElem, typename TDim, sycl_copy_type ctype>
        struct TaskCopySycl;

        //#############################################################################
        //! The host-to-device SYCL memory copy trait.
        template <typename TElem, typename TDim>
        struct TaskCopySycl<TElem, TDim, sycl_copy_type::h2d>
        {
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                auto dst_acc = dst_buf.template get_access<cl::sycl::access::mode::write>(cgh, range);
                cgh.copy(src_ptr, dst_acc);
            }

            const TElem * const src_ptr;
            cl::sycl::buffer<TElem, TDim::value> dst_buf;
            cl::sycl::range<TDim::value> range;
        };

        //#############################################################################
        //! The device-to-host SYCL memory copy trait.
        template <typename TElem, typename TDim>
        struct TaskCopySycl<TElem, TDim, sycl_copy_type::d2h>
        {
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                auto src_acc = src_buf.template get_access<cl::sycl::access::mode::read>(cgh, range);
                cgh.copy(src_acc, dst_ptr);
            }

            cl::sycl::buffer<TElem, TDim::value> src_buf;
            TElem * const dst_ptr;
            cl::sycl::range<TDim::value> range;
        };

        //#############################################################################
        //! The device-to-device SYCL memory copy trait.
        template <typename TElem, typename TDim>
        struct TaskCopySycl<TElem, TDim, sycl_copy_type::d2d>
        {
            auto operator()(cl::sycl::handler& cgh) -> void
            {
                auto src_acc = src_buf.template get_access<cl::sycl::access::mode::read>(cgh, range);
                auto dst_acc = dst_buf.template get_access<cl::sycl::access::mode::write>(cgh, range);

                cgh.copy(src_acc, dst_acc);
            }

            cl::sycl::buffer<TElem, TDim::value> src_buf;
            cl::sycl::buffer<TElem, TDim::value> dst_buf;
            cl::sycl::range<TDim::value> range;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        //#############################################################################
        //! The SYCL to CPU memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevCpu, DevSycl>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc & viewSrc, TExtent const & extent)
            {
                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, Elem<std::remove_const_t<TViewSrc>>>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return detail::TaskCopySycl<Elem<std::remove_const_t<TViewSrc>>, TDim, detail::sycl_copy_type::d2h>{
                    viewSrc.m_buf, getPtrNative(viewDst), detail::get_sycl_range<Dim<TExtent>::value>(extent)
                };
            }
        };

        //#############################################################################
        //! The CPU to SYCL memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevSycl, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc & viewSrc, TExtent const & extent)
            {
                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, Elem<std::remove_const_t<TViewSrc>>>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return detail::TaskCopySycl<Elem<std::remove_const_t<TViewSrc>>, TDim, detail::sycl_copy_type::h2d{
                    getPtrNative(viewSrc), viewDst.m_buf, detail::get_sycl_range<Dim<TExtent>::value>(extent)
                };
            }
        };

        //#############################################################################
        //! The SYCL to SYCL memory copy trait specialization.
        template<typename TDim>
        struct CreateTaskMemcpy<TDim, DevSycl, DevSycl>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc & viewSrc, TExtent const & extent)
            {
                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, Elem<std::remove_const_t<TViewSrc>>>,
                              "The source and the destination view are required to have the same element type!");
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                return detail::TaskCopySycl<Elem<std::remove_const_t<TViewSrc>>, TDim, detail::sycl_copy_type::d2d{
                    viewSrc.m_buf, viewDst.m_buf, detail::get_sycl_range<Dim<TExtent>::value>(extent)
                };
            }
        };
    }
}

#endif
