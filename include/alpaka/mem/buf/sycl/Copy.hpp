/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/elem/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/AccessorGenericSycl.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/core/Sycl.hpp>

#include <sycl/sycl.hpp>

#include <memory>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace alpaka
{
    namespace detail
    {
        template <typename TSrc, typename TDst>
        struct TaskCopySyclImpl
        {
            TaskCopySyclImpl(TSrc const src, TDst dst) : m_src{src}, m_dst{dst}
            {
            }

            TaskCopySyclImpl(TaskCopySyclImpl const&) = delete;
            auto operator=(TaskCopySyclImpl const&) -> TaskCopySyclImpl& = delete;
            TaskCopySyclImpl(TaskCopySyclImpl&&) = default;
            auto operator=(TaskCopySyclImpl&&) -> TaskCopySyclImpl& = default;
            ~TaskCopySyclImpl() = default;

            TSrc m_src;
            TDst m_dst;
            std::vector<sycl::event> dependencies = {};
            std::shared_mutex mutex{};
        };

        //#############################################################################
        //! The SYCL memory copy trait.
        template <typename TSrc, typename TDst>
        struct TaskCopySycl
        {
            auto operator()(sycl::handler& cgh) -> void
            {
                cgh.depends_on(pimpl->dependencies);

                if constexpr(!std::is_pointer_v<TSrc>)
                    cgh.require(pimpl->m_src);

                if constexpr(!std::is_pointer_v<TDst>)
                    cgh.require(pimpl->m_dst);

                cgh.copy(pimpl->m_src, pimpl->m_dst);
            }

            std::shared_ptr<TaskCopySyclImpl<TSrc, TDst>> pimpl;
            // Distinguish from non-alpaka types (= host tasks)
            static constexpr auto is_sycl_enqueueable = true;
        };
    }

    //-----------------------------------------------------------------------------
    // Trait specializations for CreateTaskMemcpy.
    namespace traits
    {
        //! The SYCL host-to-device memory copy trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltf>, DevCpu>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
            {
                using SrcType = Elem<std::remove_const_t<TViewSrc>>;
                constexpr auto SrcBytes = sizeof(SrcType);

                static_assert(!std::is_const<TViewDst>::value, "The destination view cannot be const!");

                static_assert(Dim<TViewDst>::value == Dim<std::remove_const_t<TViewSrc>>::value,
                              "The source and the destination view are required to have the same dimensionality!");

                static_assert(Dim<TViewDst>::value == Dim<TExtent>::value,
                              "The views and the extent are required to have the same dimensionality!");

                static_assert(std::is_same_v<Elem<TViewDst>, std::remove_const_t<SrcType>>,
                              "The source and the destination view are required to have the same element type!");

                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                using SrcHandle = decltype(getPtrNative(viewSrc));
                using DstHandle = decltype(accessWith<WriteAccess>(viewDst).m_acc);
                using TaskType = alpaka::detail::TaskCopySycl<SrcHandle, DstHandle>;
                using ImplType = alpaka::detail::TaskCopySyclImpl<SrcHandle, DstHandle>;

                return TaskType{std::make_shared<ImplType>(getPtrNative(viewSrc), accessWith<WriteAccess>(viewDst).m_acc)};
            }
        };

        //#############################################################################
        //! The SYCL device-to-host memory copy trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemcpy<TDim, DevCpu, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
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

                using SrcHandle = decltype(accessWith<ReadAccess>(viewSrc).m_acc);
                using DstHandle = decltype(getPtrNative(viewDst));
                using TaskType = alpaka::detail::TaskCopySycl<SrcHandle, DstHandle>;
                using ImplType = alpaka::detail::TaskCopySyclImpl<SrcHandle, DstHandle>;

                return TaskType{std::make_shared<ImplType>(accessWith<ReadAccess>(viewSrc).m_acc, getPtrNative(viewDst))};
            }
        };

        //#############################################################################
        //! The SYCL device-to-device memory copy trait specialization.
        template<typename TDim, typename TPltfDst, typename TPltfSrc>
        struct CreateTaskMemcpy<TDim, DevGenericSycl<TPltfDst>, DevGenericSycl<TPltfSrc>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TViewSrc, typename TViewDst>
            ALPAKA_FN_HOST static auto createTaskMemcpy(TViewDst & viewDst, TViewSrc const& viewSrc, TExtent const & ext)
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

                using SrcHandle = decltype(accessWith<ReadAccess>(viewSrc).m_acc);
                using DstHandle = decltype(accessWith<WriteAccess>(viewDst).m_acc);
                using TaskType = alpaka::detail::TaskCopySycl<SrcHandle, DstHandle>;
                using ImplType = alpaka::detail::TaskCopySyclImpl<SrcHandle, DstHandle>;

                return TaskType{std::make_shared<ImplType>(accessWith<ReadAccess>(viewSrc).m_acc, accessWith<WriteAccess>(viewDst).m_acc)};
            }
        };
    }
}

#endif
