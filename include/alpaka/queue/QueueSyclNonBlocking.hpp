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

#include <alpaka/dev/DevSycl.hpp>

#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    class EventSycl;

    //#############################################################################
    //! The SYCL non-blocking queue.
    class QueueSyclNonBlocking final
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueSyclNonBlocking(DevSycl & dev)
        : m_dev{dev}
        {}
        //-----------------------------------------------------------------------------
        QueueSyclNonBlocking(QueueSyclNonBlocking const &) = default;
        //-----------------------------------------------------------------------------
        QueueSyclNonBlocking(QueueSyclNonBlocking &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueSyclNonBlocking const &) -> QueueSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueSyclNonBlocking &&) -> QueueSyclNonBlocking & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(QueueSyclNonBlocking const & rhs) const -> bool
        {
            return (m_dev == rhs.m_dev) && (m_event == rhs.m_event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueSyclNonBlocking const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueSyclNonBlocking() = default;

    public:
        DevSycl m_dev; //!< The device this queue is bound to.
        cl::sycl::event m_event; //!< The last event in the dependency graph.
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL non-blocking queue device type trait specialization.
        template<>
        struct DevType<QueueSyclNonBlocking>
        {
            using type = DevSycl;
        };

        //#############################################################################
        //! The SYCL non-blocking queue device get trait specialization.
        template<>
        struct GetDev<QueueSyclNonBlocking>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(QueueSyclNonBlocking const & queue) -> DevSycl
            {
                return queue.m_dev;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue event type trait specialization.
        template<>
        struct EventType<QueueSyclNonBlocking>
        {
            using type = EventSycl;
        };

        //#############################################################################
        //! The SYCL non-blocking queue enqueue trait specialization.
        template<typename TTask>
        struct Enqueue<QueueSyclNonBlocking, TTask>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueSyclNonBlocking & queue, TTask const & task) -> void
            {
                // FIXME: This is not thread-safe.
                
                // task must be a SYCL command group function object
                queue.m_event = queue.m_dev.m_Queue.submit(task);
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue test trait specialization.
        template<>
        struct Empty<QueueSyclNonBlocking>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto empty(QueueSyclNonBlocking const & queue) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // SYCL objects are reference counted, so we can just copy the queue here
                auto non_const_queue = queue;
                // check for previous events
                if(auto prev = non_const_queue.m_event.get_wait_list(); prev.empty())
                {
                    switch(non_const_queue.m_event.get_info<cl::sycl::info::event::command_execution_status>())
                    {
                        // Last event is completed
                        case cl::sycl::info::event_command_status::complete:
                            return true;

                        // Last event is submitted or running
                        default:
                            return false;
                    }
                }

                // we are still waiting for previous events
                return false;
            }
        };

        //#############################################################################
        //! The SYCL non-blocking queue thread wait trait specialization.
        //!
        //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
        template<>
        struct CurrentThreadWaitFor<QueueSyclNonBlocking>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(QueueSyclNonBlocking const & queue) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // SYCL objects are reference counted, so we can just copy the queue here
                auto non_const_queue = queue;
                non_const_queue.m_dev.m_Queue.wait_and_throw();
            }
        };
    }
}

#endif
