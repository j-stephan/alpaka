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
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#include <alpaka/core/Sycl.hpp>

#include <stdexcept>
#include <memory>
#include <functional>

namespace alpaka
{
    //#############################################################################
    //! The SYCL device event.
    template <typename TDev>
    class EventGenericSycl final
    {
        friend struct traits::GetDev<EventGenericSycl<TDev>>;
        friend struct traits::IsComplete<EventGenericSycl<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::CurrentThreadWaitFor<EventGenericSycl<TDev>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST EventGenericSycl(TDev const& dev, bool bBusyWait = true)
        : m_dev{dev}
        , m_event{}
        , m_bBusyWait{bBusyWait}
        {
            ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
        }
        //-----------------------------------------------------------------------------
        EventGenericSycl(EventGenericSycl const &) = default;
        //-----------------------------------------------------------------------------
        EventGenericSycl(EventGenericSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericSycl const &) -> EventGenericSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(EventGenericSycl &&) -> EventGenericSycl & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(EventGenericSycl const & rhs) const -> bool
        {
            return (m_event == rhs.m_event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(EventGenericSycl const & rhs) const -> bool
        {
            return !((*this) == rhs);
        }
        //-----------------------------------------------------------------------------
        ~EventGenericSycl() = default;

    private:
        TDev m_dev;
        cl::sycl::event m_event;
        bool m_bBusyWait;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device event device get trait specialization.
        template<typename TDev>
        struct GetDev<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(EventGenericSycl<TDev> const & event)-> TDev
            {
                return event.m_dev;
            }
        };

        //#############################################################################
        //! The SYCL device event test trait specialization.
        template<typename TDev>
        struct IsComplete<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto isComplete(EventGenericSycl<TDev> const & event) -> bool
            {
                using namespace cl::sycl;
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                const auto status = event.m_event.get_info<info::event::command_execution_status>();
                return (status == info::event_command_status::complete);
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL queue enqueue trait specialization.
        template<typename TDev>
        struct Enqueue<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto enqueue(QueueGenericSyclBlocking<TDev>& queue, EventGenericSycl<TDev>& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_event = queue.m_event;
            }
        };

        //#############################################################################
        //! The SYCL device event thread wait trait specialization.
        //!
        //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
        //! If the event is not enqueued to a queue the method returns immediately.
        template<typename TDev>
        struct CurrentThreadWaitFor<EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto currentThreadWaitFor(EventGenericSycl<TDev>& event)
            -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                event.m_Event.wait_and_throw();
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericSyclNonBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueGenericSyclNonBlocking<TDev>& queue, EventGenericSycl<TDev> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                queue.m_dependencies.push_back(event.m_event);
            }
        };

        //#############################################################################
        //! The SYCL queue event wait trait specialization.
        template<typename TDev>
        struct WaiterWaitFor<QueueGenericSyclBlocking<TDev>, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(QueueSyclBlocking & queue, EventGenericSycl const & event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                queue.m_dependencies.push_back(event.m_event);
            }
        };

        //#############################################################################
        //! The SYCL device event wait trait specialization.
        //!
        //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
        template<typename TDev>
        struct WaiterWaitFor<TDev, EventGenericSycl<TDev>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto waiterWaitFor(TDev& dev, EventGenericSycl<TDev> const& event) -> void
            {
                ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                dev.m_dependencies.push_back(event.m_event);
            }
        };
    }
}

#endif
