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
                bool bBusyWait = true) :
            : m_Device{dev.m_Device}
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

                    event.
                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_CudaEvent,
                        queue.m_spQueueImpl->m_CudaQueue));
                }
            };
            //#############################################################################
            //! The CUDA RT queue enqueue trait specialization.
            template<>
            struct Enqueue<
                queue::QueueCudaRtBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto enqueue(
                    queue::QueueCudaRtBlocking & queue,
                    event::EventCudaRt & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaEventRecord(
                        event.m_spEventImpl->m_CudaEvent,
                        queue.m_spQueueImpl->m_CudaQueue));
                }
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device event thread wait trait specialization.
            //!
            //! Waits until the event itself and therefore all tasks preceding it in the queue it is enqueued to have been completed.
            //! If the event is not enqueued to a queue the method returns immediately.
            template<>
            struct CurrentThreadWaitFor<
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Sync is allowed even for events on non current device.
                    ALPAKA_CUDA_RT_CHECK(cudaEventSynchronize(
                        event.m_spEventImpl->m_CudaEvent));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaRtNonBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaRtNonBlocking & queue,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaQueue,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT queue event wait trait specialization.
            template<>
            struct WaiterWaitFor<
                queue::QueueCudaRtBlocking,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    queue::QueueCudaRtBlocking & queue,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        queue.m_spQueueImpl->m_CudaQueue,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
            //#############################################################################
            //! The CUDA RT device event wait trait specialization.
            //!
            //! Any future work submitted in any queue of this device will wait for event to complete before beginning execution.
            template<>
            struct WaiterWaitFor<
                dev::DevCudaRt,
                event::EventCudaRt>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto waiterWaitFor(
                    dev::DevCudaRt & dev,
                    event::EventCudaRt const & event)
                -> void
                {
                    ALPAKA_DEBUG_MINIMAL_LOG_SCOPE;

                    // Set the current device.
                    ALPAKA_CUDA_RT_CHECK(
                        cudaSetDevice(
                            dev.m_iDevice));

                    ALPAKA_CUDA_RT_CHECK(cudaStreamWaitEvent(
                        nullptr,
                        event.m_spEventImpl->m_CudaEvent,
                        0));
                }
            };
        }
    }
}

#endif
