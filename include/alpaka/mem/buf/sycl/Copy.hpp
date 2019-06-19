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
    namespace mem
    {
        namespace view
        {
            namespace sycl
            {
                namespace detail
                {
                    //#############################################################################
                    //! The SYCL memory copy type.
                    enum class copy_type
                    {
                        host_to_device,
                        device_to_host,
                        device_to_device
                    };

                    //#############################################################################
                    //! The SYCL memory copy trait.
                    template<
                        typename TElem,
                        typename TDim,
                        copy_type ctype>
                    struct TaskCopySycl;

                    //#############################################################################
                    //! The host-to-device SYCL memory copy trait.
                    template <typename TElem, typename TDim>
                    struct TaskCopySycl<TElem, TDim, copy_type::host_to_device>
                    {
                        auto operator()(cl::sycl::handler& cgh) -> void
                        {
                            auto dst_acc = dst_buf.template get_access<
                                cl::sycl::access::mode::write,
                                cl::sycl::access::target::global_buffer>(cgh, range);
                            cgh.copy(src_ptr, dst_acc);
                        }

                        const TElem * const src_ptr;
                        cl::sycl::buffer<TElem, TDim::value> dst_buf;
                        cl::sycl::range<TDim::value> range;
                    };

                    //#############################################################################
                    //! The device-to-host SYCL memory copy trait.
                    template <typename TElem, typename TDim>
                    struct TaskCopySycl<TElem, TDim, copy_type::device_to_host>
                    {
                        auto operator()(cl::sycl::handler& cgh) -> void
                        {
                            auto src_acc = src_buf.template get_access<
                                            cl::sycl::access::mode::read>(cgh, range);
                            cgh.copy(src_acc, dst_ptr);
                        }

                        cl::sycl::buffer<TElem, TDim::value>& src_buf;
                        TElem * const dst_ptr;
                        cl::sycl::range<TDim::value> range;
                    };

                    //#############################################################################
                    //! The device-to-device SYCL memory copy trait.
                    template <typename TElem, typename TDim>
                    struct TaskCopySycl<TElem, TDim, copy_type::device_to_device>
                    {
                        auto operator()(cl::sycl::handler& cgh) -> void
                        {
                            auto src_acc = src_buf.template get_access<
                                            cl::sycl::access::mode::read>(cgh, range);
                            auto dst_acc = dst_buf.template get_access<
                                            cl::sycl::access::mode::write>(cgh, range);

                            cgh.copy(src_acc, dst_acc);
                        }

                        cl::sycl::buffer<TElem, TDim::value>& src_buf;
                        cl::sycl::buffer<TElem, TDim::value>& dst_buf;
                        cl::sycl::range<TDim::value> range;
                    };
                }
            }
            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            namespace traits
            {
                //#############################################################################
                //! The SYCL to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc & viewSrc,
                        TExtent const & extent)
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<std::remove_const_t<TViewSrc>>::value,
                            "The source and the destination view are required to have the same dimensionality!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");

                        static_assert(
                            std::is_same_v<elem::Elem<TViewDst>, elem::Elem<std::remove_const_t<TViewSrc>>>,
                            "The source and the destination view are required to have the same element type!");

                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return mem::view::sycl::detail::TaskCopySycl<
                            elem::Elem<std::remove_const_t<TViewSrc>>,
                            TDim,
                            mem::view::sycl::detail::copy_type::device_to_host>{
                                // This is really dumb - buffers can never be const in SYCL, even when
                                // in read-only mode
                                *(const_cast<std::remove_const_t<decltype(viewSrc.m_buf)>*>(&viewSrc.m_buf)),
                                mem::view::getPtrNative(viewDst),
                                mem::view::sycl::detail::get_sycl_range<dim::Dim<TExtent>::value>(extent)
                            };
                    }
                };

                //#############################################################################
                //! The CPU to SYCL memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevSycl,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc & viewSrc,
                        TExtent const & extent)
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<std::remove_const_t<TViewSrc>>::value,
                            "The source and the destination view are required to have the same dimensionality!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");

                        static_assert(
                            std::is_same_v<elem::Elem<TViewDst>, elem::Elem<std::remove_const_t<TViewSrc>>>,
                            "The source and the destination view are required to have the same element type!");

                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return mem::view::sycl::detail::TaskCopySycl<
                            elem::Elem<std::remove_const_t<TViewSrc>>,
                            TDim,
                            mem::view::sycl::detail::copy_type::host_to_device>{
                                mem::view::getPtrNative(viewSrc),
                                viewDst.m_buf,
                                mem::view::sycl::detail::get_sycl_range<dim::Dim<TExtent>::value>(extent)
                            };
                    }
                };

                //#############################################################################
                //! The SYCL to SYCL memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevSycl,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc & viewSrc,
                        TExtent const & extent)
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");

                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return mem::view::sycl::detail::TaskCopySycl<
                            elem::Elem<TViewSrc>,
                            TDim,
                            mem::view::sycl::detail::copy_type::device_to_device>{
                                viewSrc.m_buf,
                                viewDst.m_buf,
                                mem::view::sycl::detail::get_sycl_range<dim::Dim<TExtent>::value>(extent)
                            };
                    }
                };
            }
        }
    }
}

#endif
