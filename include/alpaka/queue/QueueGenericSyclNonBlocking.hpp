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
#include <alpaka/event/Traits.hpp>
#include <alpaka/event/EventGenericSycl.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>
#include <alpaka/core/Sycl.hpp>

#include <CL/sycl.hpp>

#include <algorithm>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace alpaka
{
    //#############################################################################
    //! The SYCL non-blocking queue.
    template <typename TDev>
    class QueueGenericSyclNonBlocking final
    {
        friend struct traits::GetDev<QueueGenericSyclNonBlocking<TDev>>;
        friend struct traits::Empty<QueueGenericSyclNonBlocking<TDev>>;
        template <typename TTask> friend struct traits::Enqueue<QueueGenericSyclNonBlocking<TDev>, TTask>;
        friend struct traits::CurrentThreadWaitFor<QueueGenericSyclNonBlocking<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
        friend struct traits::WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueGenericSyclNonBlocking(TDev& dev) : m_dev{dev}
        {}
        //-----------------------------------------------------------------------------
        QueueGenericSyclNonBlocking(QueueGenericSyclNonBlocking const &) = default;
        //-----------------------------------------------------------------------------
        QueueGenericSyclNonBlocking(QueueGenericSyclNonBlocking &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclNonBlocking const &) -> QueueGenericSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclNonBlocking &&) -> QueueGenericSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(QueueGenericSyclNonBlocking const & rhs) const -> bool
        {
            return (m_dev == rhs.m_dev) && (m_event == rhs.m_event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueGenericSyclNonBlocking const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueGenericSyclNonBlocking() = default;

    private:
        TDev m_dev; //!< The device this queue is bound to.
        cl::sycl::event m_event{}; //!< The last event in the dependency graph.
        std::vector<cl::sycl::event> m_dependencies = {}; //!< A list of events this queue should wait for.
        std::shared_ptr<std::shared_mutex> mutable mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL non-blocking queue device type trait specialization.
        template<typename TDev>
        struct DevType<QueueGenericSyclNonBlocking<TDev>>
        {
            using type = TDev;
        };

        //#############################################################################
        //! The SYCL non-blocking queue device get trait specialization.
        template<typename TDev>
        struct GetDev<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(QueueGenericSyclNonBlocking<TDev> const& queue)
            {
                auto&& lock = std::shared_lock{*queue.mutex_ptr};
                return queue.m_dev;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue event type trait specialization.
        template<typename TDev>
        struct EventType<QueueGenericSyclNonBlocking<TDev>>
        {
            using type = EventGenericSycl<TDev>;
        };

        //#############################################################################
        //! The SYCL non-blocking queue enqueue trait specialization.
        template<typename TDev, typename TTask>
        struct Enqueue<QueueGenericSyclNonBlocking<TDev>, TTask>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclNonBlocking<TDev>& queue, TTask& task) -> void
            {
                auto remove_completed = [](std::vector<cl::sycl::event>& events)
                {
                    std::remove_if(begin(events), end(events), [](cl::sycl::event const& ev)
                    {
                        return (ev.get_info<info::event::command_execution_status>() == info::event_command_status::complete);
                    });
                };

                auto&& lock = std::scoped_lock{*queue.mutex_ptr, , *queue.m_dev.mutex_ptr, *task.mutex_ptr};

                // Remove any completed events from the task's dependencies
                if(!task.m_dependencies.empty())
                    remove_completed(task.m_dependencies);

                // Remove any completed events from the device's dependencies
                if(!queue.m_dev.m_dependencies.empty())
                    remove_completed(queue.m_dev.m_dependencies);
                
                // Wait for the remaining uncompleted events the device is supposed to wait for
                if(!queue.m_dev.m_dependencies.empty())
                    std::copy(begin(queue.m_dev.m_dependencies), end(queue.m_dev.m_dependencies), std::back_inserter(task.m_dependencies));
                
                // Wait for any events this queue is supposed to wait for
                if(!queue.m_dependencies.empty())
                    task.m_dependencies.insert(end(task.m_dependencies), begin(queue.m_dependencies), end(queue.m_dependencies));

                // Wait for any previous kernels running in this queue
                task.m_dependencies.push_back(queue.m_event);

                // Execute the kernel
                queue.m_event = queue.m_dev.m_Queue.submit(task);

                // Remove queue dependencies
                queue.m_dependencies.clear();
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue test trait specialization.
        template<typename TDev>
        struct Empty<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto empty(QueueGenericSyclNonBlocking<TDev> const& queue) -> bool
            {
                using namespace cl::sycl;

                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                auto&& lock = std::shared_lock{*queue.mutex_ptr};

                return queue.m_event.get_info<info::event::command_execution_status>() == info::event_command_status::complete;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
        template<typename TDev>
        struct CurrentThreadWaitFor<QueueGenericSyclNonBlocking<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueGenericSyclNonBlocking<TDev> const& queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                auto&& lock = std::unique_lock{*queue.mutex_ptr};

                // SYCL objects are reference counted, so we can just copy the queue here
                auto non_const_queue = queue;
                non_const_queue.m_dev.m_queue.wait_and_throw();
            }
        };
    }
}

#endif
