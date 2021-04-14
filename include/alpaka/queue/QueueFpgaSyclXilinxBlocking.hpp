/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#if defined(ALPAKA_ACC_SYCL_ENABLED) && defined(ALPAKA_SYCL_BACKEND_XILINX)

#include <alpaka/dev/DevFpgaSyclXilinx.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/event/Traits.hpp>
#include <alpaka/queue/QueueGenericSyclBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <sycl/sycl.hpp>

#include <memory>
#include <new>

namespace alpaka
{
    //! The SYCL blocking queue.
    template <>
    class QueueGenericSyclBlocking<DevFpgaSyclXilinx>
    {
        friend struct traits::GetDev<QueueGenericSyclBlocking<DevFpgaSyclXilinx>>;
        friend struct traits::Empty<QueueGenericSyclBlocking<DevFpgaSyclXilinx>>;
        template <typename TQueue, typename TTask, typename Sfinae> friend struct traits::Enqueue;
        friend struct traits::CurrentThreadWaitFor<QueueGenericSyclBlocking<DevFpgaSyclXilinx>>;
        friend struct traits::Enqueue<QueueGenericSyclBlocking<DevFpgaSyclXilinx>, EventGenericSycl<DevFpgaSyclXilinx>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclBlocking<DevFpgaSyclXilinx>, EventGenericSycl<DevFpgaSyclXilinx>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueGenericSyclBlocking(DevFpgaSyclXilinx const& dev)
        : m_dev{dev}
        , m_queue{dev.m_context, // This is important. In SYCL a device can belong to multiple contexts.
                  dev.m_device, {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}}
        {
            detail::jumpstart_device(m_queue);
        }
        //-----------------------------------------------------------------------------
        QueueGenericSyclBlocking(QueueGenericSyclBlocking const &) = default;
        //-----------------------------------------------------------------------------
        QueueGenericSyclBlocking(QueueGenericSyclBlocking &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclBlocking const &) -> QueueGenericSyclBlocking & = default;
        //-----------------------------------------------------------------------------
        auto operator=(QueueGenericSyclBlocking &&) -> QueueGenericSyclBlocking & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(QueueGenericSyclBlocking const & rhs) const -> bool
        {
            return (m_dev == rhs.m_dev) && (m_event == rhs.m_event);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueGenericSyclBlocking const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueGenericSyclBlocking() = default;

    private:
        DevFpgaSyclXilinx m_dev; //!< The device this queue is bound to.
        sycl::queue m_queue; //!< The underlying SYCL queue.
        sycl::event m_event{}; //!< The last event in the dependency graph.
        std::vector<sycl::event> m_dependencies = {}; //!< A list of events this queue should wait for.
        std::shared_ptr<std::shared_mutex> mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    using QueueFpgaSyclXilinxBlocking = QueueGenericSyclBlocking<DevFpgaSyclXilinx>;
}

#endif
