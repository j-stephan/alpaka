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

#include <stdexcept>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <thread>

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
        //! The SYCL async queue.
        class QueueSyclAsync final
        {
        public:
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST QueueSyclAsync(
                dev::DevSycl const & dev)
                : m_dev{dev}
            {}
            //-----------------------------------------------------------------------------
            QueueSyclAsync(QueueSyclAsync const &) = default;
            //-----------------------------------------------------------------------------
            QueueSyclAsync(QueueSyclAsync &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclAsync const &) -> QueueSyclAsync & = default;
            //-----------------------------------------------------------------------------
            auto operator=(QueueSyclAsync &&) -> QueueSyclAsync & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(QueueSyclAsync const & rhs) const -> bool = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(QueueSyclAsync const & rhs) const -> bool = default;
            //-----------------------------------------------------------------------------
            ~QueueSyclAsync() = default;

        public:
            dev::DevSycl const & m_dev; //!< The device this queue is bound to.
            std::vector<cl::sycl::event> m_events;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL async queue device type trait specialization.
            template<>
            struct DevType<
                queue::QueueSyclAsync>
            {
                using type = dev::DevSycl;
            };
            //#############################################################################
            //! The SYCL async queue device get trait specialization.
            template<>
            struct GetDev<
                queue::QueueSyclAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    queue::QueueSyclAsync const & queue)
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
            //! The SYCL async queue event type trait specialization.
            template<>
            struct EventType<
                queue::QueueSyclAsync>
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
                queue::QueueSyclAsync,
                TTask>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueSyclAsync & queue,
                    TTask const & task)
                -> void
                {
                    // task must be a SYCL command group function object
                    auto event = queue.m_dev.m_Queue.submit(task);
                    queue.m_events.push_back(std::move(event));
                }
            };

            //#############################################################################
            //! The SYCL async queue test trait specialization.
            template<>
            struct Empty<
                queue::QueueSyclAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto empty(
                    queue::QueueSyclAsync const & queue)
                -> bool
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;
                    return queue.m_events.empty();
                }
            };
        }
    }

    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL async queue thread wait trait specialization.
            //!
            //! Blocks execution of the calling thread until the queue has finished processing all previously requested tasks (kernels, data copies, ...)
            template<>
            struct CurrentThreadWaitFor<
                queue::QueueSyclAsync>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    queue::QueueSyclAsync const & queue)
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
