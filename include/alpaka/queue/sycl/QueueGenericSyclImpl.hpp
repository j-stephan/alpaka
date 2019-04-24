/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <CL/sycl.hpp>

#include <algorithm>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <type_traits>
#include <vector>

namespace alpaka::experimental::detail
{
    template<typename T, typename = void>
    struct is_sycl_task : std::false_type
    {
    };

    template<typename T>
    struct is_sycl_task<T, std::void_t<decltype(T::is_sycl_task)>> : std::true_type
    {
    };

    template<typename T, typename = void>
    struct is_sycl_kernel : std::false_type
    {
    };

    template<typename T>
    struct is_sycl_kernel<T, std::void_t<decltype(T::is_sycl_kernel)>> : std::true_type
    {
    };

    class QueueGenericSyclImpl
    {
    public:
        QueueGenericSyclImpl(sycl::context context, sycl::device device)
            : m_queue{
                context, // This is important. In SYCL a device can belong to multiple contexts.
                device,
                {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}}
        {
        }

        // Don't call this without locking first!
        auto clean_dependencies()
        {
            // Clean up completed events
            if(!m_dependencies.empty())
            {
                auto const start = std::begin(m_dependencies);
                auto const old_end = std::end(m_dependencies);
                auto const new_end = std::remove_if(
                    start,
                    old_end,
                    [](sycl::event ev) {
                        return ev.get_info<sycl::info::event::command_execution_status>()
                            == sycl::info::event_command_status::complete;
                    });

                m_dependencies.erase(new_end, old_end);
            }
        }

        auto register_dependency(sycl::event event)
        {
            std::lock_guard<std::shared_mutex> lock{m_mutex};

            clean_dependencies();
            m_dependencies.push_back(event);
        }

        auto empty() const
        {
            std::shared_lock<std::shared_mutex> lock{m_mutex};
            return m_last_event.get_info<sycl::info::event::command_execution_status>()
                == sycl::info::event_command_status::complete;
        }

        auto wait()
        {
            std::lock_guard<std::shared_mutex> lock{m_mutex};
            m_queue.wait_and_throw();
        }

        auto get_last_event() const
        {
            std::shared_lock<std::shared_mutex> lock{m_mutex};
            return m_last_event;
        }

        template<typename TTask>
        auto enqueue(TTask const& task, bool wait)
        {
            std::lock_guard<std::shared_mutex> lock{m_mutex};

            clean_dependencies();

            // Execute task
            m_last_event = m_queue.submit(
                [this, &task](sycl::handler& cgh)
                {
                    if(!m_dependencies.empty())
                        cgh.depends_on(m_dependencies);

                    if constexpr(is_sycl_kernel<TTask>()) // Kernel
                        task(cgh, fence_dummy);
                    else if constexpr(is_sycl_task<TTask>()) // Copy / Fill
                        task(cgh); // Will call cgh.parallel_for (or copy, fill, ...) internally
                    else // Host
                        cgh.host_task(task);
                });

            m_dependencies.clear();

            if(wait)
                m_queue.wait_and_throw();
        }

        sycl::queue m_queue;
        std::vector<sycl::event> m_dependencies = {};
        sycl::event m_last_event = {};
        sycl::buffer<int, 1> fence_dummy{sycl::range<1>{1}};
        std::shared_mutex mutable m_mutex;
    };
} // namespace alpaka::experimental::detail
