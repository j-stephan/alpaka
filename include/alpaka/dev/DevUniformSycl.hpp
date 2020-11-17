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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    //#############################################################################
    //! The SYCL device handle.
    class DevUniformSycl : public concepts::Implements<ConceptCurrentThreadWaitFor, DevUniformSycl>
    {
        friend struct traits::GetDevByIdx<PltfUniformSycl>;
        friend struct traits::GetName<DevUniformSycl>;
        friend struct traits::GetMemBytes<DevUniformSycl>;

    protected:
        //-----------------------------------------------------------------------------
        DevUniformSycl() = default;
    public:
        DevUniformSycl(cl::sycl::device device, cl::sycl::context context, cl::sycl::queue queue)
        : m_device{device}, m_context{context}, m_queue{queue}
        {}

        //-----------------------------------------------------------------------------
        DevUniformSycl(DevUniformSycl const &) = default;
        //-----------------------------------------------------------------------------
        DevUniformSycl(DevUniformSycl &&) = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevUniformSycl const &) -> DevUniformSycl & = default;
        //-----------------------------------------------------------------------------
        auto operator=(DevUniformSycl &&) -> DevUniformSycl & = default;
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator==(DevUniformSycl const & rhs) const -> bool
        {
            return (rhs.m_Device == m_Device) && (rhs.m_Queue == m_Queue);
        }
        //-----------------------------------------------------------------------------
        ALPAKA_FN_HOST auto operator!=(DevUniformSycl const & rhs) const -> bool
        {
            return !operator==(rhs);
        }
        //-----------------------------------------------------------------------------
        ~DevUniformSycl() = default;

    protected:
        cl::sycl::device m_device;
        cl::sycl::context m_context;
        cl::sycl::queue m_queue;
    };

    namespace traits
    {
        //#############################################################################
        //! The SYCL device name get trait specialization.
        template<typename TDev>
        struct GetName<TDev, std::enable_if_t<std::is_base_of<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getName(TDev const& dev) -> std::string
            {
                return dev.m_device.get_info<cl::sycl::info::device::name>();
            }
        };

        //#############################################################################
        //! The SYCL device available memory get trait specialization.
        template<typename TDev>
        struct GetMemBytes<TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getMemBytes(TDev const& dev) -> std::size_t
            {
                return dev.m_device.get_info<cl::sycl::info::device::global_mem_size>();
            }
        };

        //#############################################################################
        //! The SYCL device free memory get trait specialization. Note that
        //! this function will usually return the size of the device memory
        //! as there is no standard way in SYCL to query free memory.
        template<typename TDev>
        struct GetFreeMemBytes<TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getFreeMemBytes(TDev const& dev) -> std::size_t
            {
                // There is no way in SYCL to query free memory. If you find a way be sure to update the
                // documentation above.
                std::cerr << "[SYCL] Warning: Querying free device memory not supported for SYCL devices.\n";
                return getMemBytes(dev);
            }
        };

        //#############################################################################
        //! The SYCL device reset trait specialization. Note that this
        //! function won't actually do anything. If you need to reset your
        //! SYCL device its destructor must be called.
        template<typename TDev>
        struct Reset<TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto reset(TDev const&) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                std::cerr << "[SYCL] Warning: Explicit device reset not supported for SYCL devices\n";
            }
        };

        //#############################################################################
        //! The SYCL device memory buffer type trait specialization.
        template<typename TElem, typename TDim, typename TIdx>
        struct BufType<DevUniformSycl, TElem, TDim, TIdx>
        {
            using type = BufUniformSycl<TElem, TDim, TIdx>;
        };

        //#############################################################################
        //! The SYCL device platform type trait specialization.
        template<>
        struct PltfType<DevUniformSycl>
        {
            using type = PltfUniformSycl;
        };

        //#############################################################################
        //! The thread SYCL device wait specialization.
        //!
        //! Blocks until the device has completed all preceding requested tasks.
        //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
        template<typename TDev>
        struct CurrentThreadWaitFor<TDev, std::enable_if_t<std::is_base_of_v<DevUniformSycl, TDev>>>
        {
            ALPAKA_FN_HOST static auto currentThreadWaitFor(TDev const&) -> void
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;
                std::cerr << "[SYCL] Warning: You cannot wait for SYCL devices. Use the queue instead.\n";
            }
        };
    }
}

#endif
