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
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/QueueSyclNonBlocking.hpp>
#include <alpaka/queue/QueueSyclBlocking.hpp>
#include <alpaka/core/Sycl.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    //#############################################################################
    //! The SYCL device event.
    class EventSycl final
    {
    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST EventSycl(DevSycl const & dev, bool bBusyWait = true)
        : m_Device{dev}
        , m_Event{}
        , m_bBusyWait{bBusyWait}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }
        //-----------------------------------------------------------------------------
        EventSycl(EventSycl const &) = default;
        //-----------------------------------------------------------------------------
        EventSycl(EventSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventSycl const &) -> EventSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventSycl &&) -> EventSycl & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(EventSycl const & rhs) const -> bool
        {
            return (m_Event == rhs.m_Event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(EventSycl const & rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~EventSycl() = default;

    public:
        DevSycl m_Device;
        cl::sycl::event m_Event;
        bool m_bBusyWait;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device event device get trait specialization.
        template<>
        struct GetDev<EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(EventSycl const & event)-> DevSycl
            {
                return event.m_Device;
            }
        };

        //#############################################################################
        //! The SYCL device event test trait specialization.
        template<>
        struct IsComplete<EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isComplete(EventSycl const & event) -> bool
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                const auto status = event.m_Event.get_info<cl::sycl::info::event::command_execution_status>();
                return (status == cl::sycl::info::event_command_status::complete);
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueSyclNonBlocking, EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueSyclNonBlocking & queue, EventSycl & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_Event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<>
        struct Enqueue<QueueSyclBlocking, EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueSyclBlocking & queue, EventSycl & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_Event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
        //! If the event is not enqueued to a queue the method returns immediately.
        template<>
        struct CurrentThreadWaitFor<EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventSycl & event)
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_Event.wait_and_throw();
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueSyclNonBlocking, EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueSyclNonBlocking & queue, EventSycl const & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // FIXME: This function makes no sense in SYCL. Dependency 
                // synchronization works via buffer availability (done by
                // the SYCL runtime), not the completion of events. We could
                // emulate the behaviour using SYCL events but this would
                // require a lot of work with regard to thread-safety.
                // Additionally, we still wouldn't have the guarantee that
                // the desired memory is actually available.
                std::cerr
                    << "[SYCL] Warning: queues cannot wait for events. Blocking thread."
                    << std::endl;
                auto non_const_event = event;
                non_const_event.m_Event.wait_and_throw();
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<>
        struct WaiterWaitFor<QueueSyclBlocking, EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueSyclBlocking & queue, EventSycl const & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                // FIXME: This function makes no sense in SYCL. Dependency 
                // synchronization works via buffer availability (done by
                // the SYCL runtime), not the completion of events. We could
                // emulate the behaviour using SYCL events but this would
                // require a lot of work with regard to thread-safety.
                // Additionally, we still wouldn't have the guarantee that
                // the desired memory is actually available.
                std::cerr
                    << "[SYCL] Warning: queues cannot wait for events. Blocking thread."
                    << std::endl;
                auto non_const_event = event;
                non_const_event.m_Event.wait_and_throw();
            }
        };

        //#############################################################################
        //! The SYCL device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
        template<>
        struct WaiterWaitFor<DevSycl, EventSycl>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(DevSycl & dev, EventSycl const & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                // FIXME: This function makes no sense in SYCL. Dependency 
                // synchronization works via buffer availability (done by
                // the SYCL runtime), not the completion of events. We could
                // emulate the behaviour using SYCL events but this would
                // require a lot of work with regard to thread-safety.
                // Additionally, we still wouldn't have the guarantee that
                // the desired memory is actually available.
                std::cerr
                    << "[SYCL] Warning: devices cannot wait for events. Blocking thread."
                    << std::endl;
                auto non_const_event = event;
                non_const_event.m_Event.wait_and_throw();
            }
        };
    }
}

#endif
