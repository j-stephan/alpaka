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
    namespace event
    {
        //#############################################################################
        //! The SYCL device event.
        class EventSycl final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST EventSycl(
                dev::DevSycl const & dev,
                bool bBusyWait = true)
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
            ALPAKA_FN_HOST auto operator==(EventSycl const & rhs) const
            -> bool
            {
                return (m_Event == rhs.m_Event);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(EventSycl const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~EventSycl() = default;

        public:
            dev::DevSycl m_Device;
            cl::sycl::event m_Event;
            bool m_bBusyWait;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL device event device get trait specialization.
            template<>
            struct GetDev<
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    event::EventSycl const & event)
                -> dev::DevSycl
                {
                    return event.m_Device;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL device event test trait specialization.
            template<>
            struct Test<
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto test(
                    event::EventSycl const & event)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    const auto status = event.m_Event.get_info<
                        cl::sycl::info::event::command_execution_status>();

                    return (status == cl::sycl::info::event_command_status::complete);
                }
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueSyclNonBlocking,
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclNonBlocking & queue,
                    event::EventSycl & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    event.m_Event = queue.m_event;
                }
            };
            //#############################################################################
            //! The SYCL queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueSyclBlocking,
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclBlocking & queue,
                    event::EventSycl & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    event.m_Event = queue.m_event;
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventSycl & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    event.m_Event.wait_and_throw();
                }
            };
            //#############################################################################
            //! The SYCL queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueSyclNonBlocking,
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueSyclNonBlocking & queue,
                    event::EventSycl const & event)
                -> void
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
                        << "[SYCL] Warning: queues can't wait for events. Blocking thread."
                        << std::endl;
                    auto non_const_event = event;
                    non_const_event.m_Event.wait_and_throw();
                }
            };
            //#############################################################################
            //! The SYCL queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueSyclBlocking,
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueSyclBlocking & queue,
                    event::EventSycl const & event)
                -> void
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
                        << "[SYCL] Warning: queues can't wait for events. Blocking thread."
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
            struct WaiterWaitFor<
                dev::DevSycl,
                event::EventSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevSycl & dev,
                    event::EventSycl const & event)
                -> void
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
                        << "[SYCL] Warning: devices can't wait for events. Blocking thread."
                        << std::endl;
                    auto non_const_event = event;
                    non_const_event.m_Event.wait_and_throw();
                }
            };
        }
    }
}

#endif
