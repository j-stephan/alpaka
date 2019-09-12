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

#include <mutex>

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
    namespace event
    {
        class EventSycl;
    }
}

namespace alpaka
{
    namespace queue
    {
        //#############################################################################
        //! The SYCL blocking queue.
        class QueueSyclBlocking final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueSyclBlocking(
                dev::DevSycl dev)
                : m_dev{dev}
            {}
            //-----------------------------------------------------------------------------
            QueueSyclBlocking(QueueSyclBlocking const &) = default;
            //-----------------------------------------------------------------------------
            QueueSyclBlocking(QueueSyclBlocking &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclBlocking const &) -> QueueSyclBlocking & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclBlocking &&) -> QueueSyclBlocking & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueSyclBlocking const & rhs) const -> bool
            {
                return (m_dev == rhs.m_dev) && (m_event == rhs.m_event);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueSyclBlocking const & rhs) const -> bool
            {
                return !operator==(rhs);
            }
            //-----------------------------------------------------------------------------
            ~QueueSyclBlocking() = default;

        public:
            dev::DevSycl m_dev; //!< The device this queue is bound to.
            cl::sycl::event m_event; //!< The last event in the dependency graph.
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL blocking queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueSyclBlocking>
            {
                using type = dev::DevSycl;
            };
            //#############################################################################
            //! The SYCL blocking queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueSyclBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueSyclBlocking const & queue)
                -> dev::DevSycl
                {
                    return queue.m_dev;
                }
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL blocking queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueSyclBlocking>
            {
                using type = event::EventSycl;
            };
        }
    }
    namespace queue
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL blocking queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueSyclBlocking,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclBlocking & queue,
                    TTask const & task)
                -> void
                {
                    // FIXME: This is not thread-safe. But who uses the same
                    //  blocking queue in multiple threads anyway?

                    // task has to be a SYCL command group object 
                    queue.m_event = queue.m_dev.m_Queue.submit(task);
                    queue.m_dev.m_Queue.wait_and_throw();
                }
            };
            //#############################################################################
            //! The SYCL blocking queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueSyclBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueSyclBlocking const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // SYCL objects are reference counted, so we can just copy the queue here
                    auto non_const_queue = queue;
                    // check for previous events
                    if(auto prev = non_const_queue.m_event.get_wait_list();
                            prev.empty())
                    {
                        switch(non_const_queue.m_event.get_info<
                                cl::sycl::info::event::command_execution_status>())
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
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL blocking queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueSyclBlocking>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueSyclBlocking const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // SYCL objects are reference counted, so we can just copy the queue here
                    auto non_const_queue = queue;
                    non_const_queue.m_dev.m_Queue.wait_and_throw();
                }
            };
        }
    }
}

#endif
