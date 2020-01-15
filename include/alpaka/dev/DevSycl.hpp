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
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfSycl;
    }

    namespace dev
    {
        //#############################################################################
        //! The SYCL device handle.
        class DevSycl : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevSycl>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfSycl>;

        protected:
            //-----------------------------------------------------------------------------
            DevSycl() = default;
        public:
            DevSycl(cl::sycl::device device, cl::sycl::queue queue)
            : m_Device{device}, m_Queue{queue}
            {}
            //-----------------------------------------------------------------------------
            DevSycl(DevSycl const &) = default;
            //-----------------------------------------------------------------------------
            DevSycl(DevSycl &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevSycl const &) -> DevSycl & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevSycl &&) -> DevSycl & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevSycl const & rhs) const -> bool
            {
                return (rhs.m_Device == m_Device) && (rhs.m_Queue == m_Queue);
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevSycl const & rhs) const -> bool
            {
                return !operator==(rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevSycl() = default;

        public:
            cl::sycl::device m_Device;
            cl::sycl::queue m_Queue;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL device name get trait specialization.
            template<>
            struct GetName<
                dev::DevSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevSycl const & dev)
                -> std::string
                {
                    // get_info returns std::string in this case
                    return dev.m_Device.get_info<
                        cl::sycl::info::device::name>();
                }
            };

            //#############################################################################
            //! The SYCL device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevSycl const & dev)
                -> std::size_t
                {
                    // get_info returns cl_ulong in this case
                    return dev.m_Device.get_info<
                        cl::sycl::info::device::global_mem_size>();
                }
            };

            //#############################################################################
            //! The SYCL device free memory get trait specialization. Note that
            //! this function will usually return the size of the device memory
            //! as there is no standard way in SYCL or OpenCL to query free
            //! memory. Accelerators supporting the
            //! 'cl_amd_device_attribute_query' extension will return the amount
            //! of free memory.
            template<>
            struct GetFreeMemBytes<
                dev::DevSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevSycl const & dev)
                -> std::size_t
                {
                    // FIXME: The Intel compiler doesn't like the AMD extension
                    /*
                    if(dev.m_Device.has_extension("cl_amd_device_attribute_query"))
                    {
                        auto kiB = cl_ulong{};
                        if(auto err = clGetDeviceInfo(dev.m_Device.get(),
                                                      CL_DEVICE_GLOBAL_FREE_MEMORY_AMD,
                                                      sizeof(cl_ulong), &kiB,
                                                      nullptr);
                           err != CL_SUCCESS)
                        {
                            switch(err)
                            {
                                case CL_INVALID_DEVICE:
                                    throw std::runtime_error{"getFreeMemBytes: invalid device"};

                                case CL_INVALID_VALUE:
                                    throw std::runtime_error{"getFreeMemBytes: invalid value"};
                            }
                        }
                        return kiB * 1024ul;
                    }
                    else
                    {*/
                        // FIXME: There is no way in either SYCL or OpenCL to
                        // query free memory. If you find a way be sure to update the
                        // documentation above.
                        std::cerr <<
                            "[SYCL] Warning: Querying free device memory unsupported."
                            << std::endl;
                        return getMemBytes(dev);
                    //}
                }
            };

            //#############################################################################
            //! The SYCL device reset trait specialization. Note that this
            //! function won't actually do anything. If you need to reset your
            //! SYCL device its destructor must be called.
            template<>
            struct Reset<
                dev::DevSycl>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevSycl const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;
                    std::cerr
                        << "[SYCL] Warning: Explicit device reset not supported on SYCL platforms"
                        << std::endl;
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufSycl;

            namespace traits
            {
                //#############################################################################
                //! The SYCL device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevSycl,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufSycl<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevSycl>
            {
                using type = pltf::PltfSycl;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread SYCL device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevSycl>
            {
                //-----------------------------------------------------------------------------
                // Note the missing const behind DevSycl. wait_and_throw isn't
                // const unfortunately.
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevSycl & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    dev.m_Queue.wait_and_throw();
                }
            };
        }
    }
}

#endif
