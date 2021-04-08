/* Copyright 2020 Jan Stephan
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
#include <alpaka/queue/QueueGenericSyclNonBlocking.hpp>
#include <alpaka/queue/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <sycl/sycl.hpp>

#include <array>

namespace alpaka
{
    //! The SYCL non-blocking queue.
    template <>
    class QueueGenericSyclNonBlocking<DevFpgaSyclXilinx> final
    {
        friend struct traits::GetDev<QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>>;
        friend struct traits::Empty<QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>>;
        template <typename TQueue, typename TTask, typename Sfinae> friend struct traits::Enqueue;
        friend struct traits::CurrentThreadWaitFor<QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>>;
        friend struct traits::Enqueue<QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>, EventGenericSycl<DevFpgaSyclXilinx>>;
        friend struct traits::WaiterWaitFor<QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>, EventGenericSycl<DevFpgaSyclXilinx>>;

    public:
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST QueueGenericSyclNonBlocking(DevFpgaSyclXilinx const& dev)
        : m_dev{dev}
        , m_queue{dev.m_context, // This is important. In SYCL a device can belong to multiple contexts.
                  dev.m_device, {sycl::property::queue::enable_profiling{}, sycl::property::queue::in_order{}}}
        {
            // This is a workaround for a known bug in the Xilinx SYCL implementation:
            // https://github.com/triSYCL/sycl/issues/40. Unless we jumpstart the runtime by executing a NOOP kernel,
            // copies / memsets preceding the first kernel launch will fail because XRT fails to map them to the
            // virtual devices in sw_emu / hw_emu mode.

            constexpr auto size = 10;
            auto data = std::array<int, size>{};

            // Another issue: All kernels must have at least 1 accessor or xocc will complain.
            auto buf = sycl::buffer<int, 1>{data};

            m_queue.submit([&](sycl::handler& cgh)
            {
                auto acc = buf.get_access<sycl::access::mode::write>(cgh);
                cgh.single_task<detail::xilinx_noop_kernel<QueueGenericSyclNonBlocking>>([=]()
                {
                    acc[0] = 1;
                });
            });
            m_queue.wait_and_throw();
        }

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
            return (m_queue == rhs.m_queue);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(QueueGenericSyclNonBlocking const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~QueueGenericSyclNonBlocking() = default;

    private:
        DevFpgaSyclXilinx m_dev; //!< The device this queue is bound to.
        sycl::queue m_queue; //!< The underlying SYCL queue.
        sycl::event m_event{}; //!< The last event in the dependency graph.
        std::vector<sycl::event> m_dependencies = {}; //!< A list of events this queue should wait for.
        std::shared_ptr<std::shared_mutex> mutable mutex_ptr{std::make_shared<std::shared_mutex>()};
    };

    using QueueFpgaSyclXilinxNonBlocking = QueueGenericSyclNonBlocking<DevFpgaSyclXilinx>;
}

#endif
