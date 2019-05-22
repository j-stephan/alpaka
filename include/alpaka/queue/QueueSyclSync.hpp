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
        //! The SYCL sync queue.
        class QueueSyclSync final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueSyclSync(
                dev::DevSycl const & dev)
                : m_dev{dev}
            {}
            //-----------------------------------------------------------------------------
            QueueSyclSync(QueueSyclSync const &) = default;
            //-----------------------------------------------------------------------------
            QueueSyclSync(QueueSyclSync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclSync const &) -> QueueSyclSync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclSync &&) -> QueueSyclSync & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueSyclSync const & rhs) const -> bool = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueSyclSync const & rhs) const -> bool = default;
            //-----------------------------------------------------------------------------
            ~QueueSyclSync() = default;

        public:
            dev::DevSycl const & m_dev; //!< The device this queue is bound to.
            cl::sycl::event m_event; //!< The last event in the dependency graph.
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL sync queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueSyclSync>
            {
                using type = dev::DevSycl;
            };
            //#############################################################################
            //! The SYCL sync queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueSyclSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueSyclSync const & queue)
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
            //! The SYCL sync queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueSyclSync>
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
            //! The SYCL sync queue enqueue trait specialization.
            template<
                typename TTask>
            struct Enqueue<
                queue::QueueSycl,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclSync & queue,
                    TTask const & task)
                -> void
                {
                    // task has to be a SYCL command group object
                    queue.m_event = queue.m_dev.m_Queue.submit(task);
                    queue.m_dev.m_Queue.wait_and_throw();
                }
            };
            //#############################################################################
            //! The SYCL sync queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueSyclSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueSyclSync const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    // check for previous events
                    if(auto prev = queue.m_event.get_wait_list(); prev.empty())
                    {
                        switch(queue.m_event.get_info<
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
            //! The SYCL sync queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueSyclSync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueSyclSync const & queue)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    queue.m_dev.m_Queue.wait_and_throw();
                }
            };
        }
    }
}

#endif
