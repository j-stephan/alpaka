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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/dev/DevGenericSycl.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/AccessorGenericSycl.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <vector>

namespace alpaka
{
    namespace detail
    {
        template <typename THandle>
        struct TaskSetSyclImpl
        {
            TaskSetSyclImpl(THandle acc, std::byte val)
            : m_acc{acc}, m_val{val}
            {}

            TaskSetSyclImpl(TaskSetSyclImpl const&) = delete;
            auto operator=(TaskSetSyclImpl const&) -> TaskSetSyclImpl& = delete;
            TaskSetSyclImpl(TaskSetSyclImpl&&) = default;
            auto operator=(TaskSetSyclImpl&&) -> TaskSetSyclImpl& = default;
            ~TaskSetSyclImpl() = default;

            THandle m_acc;
            std::byte m_val;
            std::vector<sycl::event> dependencies = {};
            std::shared_mutex mutex{};
        };

        //#############################################################################
        //! The SYCL memory set trait.
        template<typename THandle>
        struct TaskSetSycl
        {
            //-----------------------------------------------------------------------------
            auto operator()(sycl::handler& cgh) -> void
            {
                cgh.depends_on(pimpl->dependencies);
                cgh.require(pimpl->m_acc);
                cgh.fill(pimpl->m_acc, pimpl->m_val);
            }

            std::shared_ptr<TaskSetSyclImpl<THandle>> pimpl;
            // Distinguish from non-alpaka types (= host tasks)
            static constexpr auto is_sycl_enqueueable = true;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemset<TDim, DevGenericSycl<TPltf>>
        {
            //-----------------------------------------------------------------------------
            template<typename TExtent, typename TView>
            ALPAKA_FN_HOST static auto createTaskMemset(TView & view, std::uint8_t const& byte, TExtent const& ext)
            {
                using Handle = sycl::accessor<std::byte, Dim<TExtent>::value, sycl::access_mode::write,
                                              sycl::access::target::global_buffer, sycl::access::placeholder::true_t>;
                using TaskType = alpaka::detail::TaskSetSycl<Handle>;
                using ImplType = alpaka::detail::TaskSetSyclImpl<Handle>;

                if constexpr(Dim<TExtent>::value == 1)
                {
                    auto const range = sycl::range<1>{static_cast<std::size_t>(extent::getWidth(ext))};
                    auto byteBuf = view.m_buf.template reinterpret<std::byte, 1>(range);
                    return TaskType{std::make_shared<ImplType>>(accessWith<WriteAccess>(byteBuf).m_acc, std::byte{byte})};
                }
                else if constexpr(Dim<TExtent>::value == 2)
                {
                    auto const range = sycl::range<2>{static_cast<std::size_t>(extent::getWidth(ext)),
                                                      static_cast<std::size_t>(extent::getHeight(ext))};
                    auto byteBuf = view.m_buf.template reinterpret<std::byte, 2>(range);
                    return TaskType{std::make_shared<ImplType>>(accessWith<WriteAccess>(byteBuf).m_acc, std::byte{byte})};
                }
                else
                {
                    auto const range = sycl::range<3>{static_cast<std::size_t>(extent::getWidth(ext)),
                                                      static_cast<std::size_t>(extent::getHeight(ext)),
                                                      static_cast<std::size_t>(extent::getDepth(ext))};
                    auto byteBuf = view.m_buf.template reinterpret<std::byte, 2>(range);
                    return TaskType{std::make_shared<ImplType>>(accessWith<WriteAccess>(byteBuf).m_acc, std::byte{byte})};
                }
            }
        };
    }
}

#endif
