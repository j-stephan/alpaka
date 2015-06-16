/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/SysInfo.hpp>      // getCpuName, getGlobalMemSizeBytes

#include <alpaka/dev/Traits.hpp>        // DevType
#include <alpaka/event/Traits.hpp>      // EventType
#include <alpaka/stream/Traits.hpp>     // StreamType
#include <alpaka/wait/Traits.hpp>       // CurrentThreadWaitFor

#include <boost/core/ignore_unused.hpp> // boost::ignore_unused

#include <map>                          // std::map
#include <sstream>                      // std::stringstream
#include <limits>                       // std::numeric_limits
#include <thread>                       // std::thread
#include <mutex>                        // std::mutex
#include <memory>                       // std::shared_ptr

namespace alpaka
{
    namespace stream
    {
        class StreamCpuAsync;

        namespace cpu
        {
            namespace detail
            {
                class StreamCpuImpl;
            }
        }
    }
    namespace dev
    {
        //-----------------------------------------------------------------------------
        //! The CPU device.
        //-----------------------------------------------------------------------------
        namespace cpu
        {
            namespace detail
            {
                //#############################################################################
                //! The CPU device implementation.
                //#############################################################################
                class DevCpuImpl
                {
                    friend stream::StreamCpuAsync;                   // StreamCpuAsync::StreamCpuAsync calls RegisterStream.
                    friend stream::cpu::detail::StreamCpuImpl;  // StreamCpuImpl::~StreamCpuImpl calls UnregisterStream.
                public:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl() = default;
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl(DevCpuImpl const &) = default;
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST DevCpuImpl(DevCpuImpl &&) = default;
                    //-----------------------------------------------------------------------------
                    //! Copy assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(DevCpuImpl const &) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Move assignment operator.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto operator=(DevCpuImpl &&) -> DevCpuImpl & = default;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST ~DevCpuImpl() = default;

                    //-----------------------------------------------------------------------------
                    //! \return The list of all streams on this device.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto GetAllStreams() const noexcept(false)
                    -> std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuImpl>>
                    {
                        std::vector<std::shared_ptr<stream::cpu::detail::StreamCpuImpl>> vspStreams;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto const & pairStream : m_mapStreams)
                        {
                            auto spStream(pairStream.second.lock());
                            if(spStream)
                            {
                                vspStreams.emplace_back(std::move(spStream));
                            }
                            else
                            {
                                throw std::logic_error("One of the streams registered on the device is invalid!");
                            }
                        }
                        return vspStreams;
                    }

                private:
                    //-----------------------------------------------------------------------------
                    //! Registers the given stream on this device.
                    //! NOTE: Every stream has to be registered for correct functionality of device wait operations!
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto RegisterStream(std::shared_ptr<stream::cpu::detail::StreamCpuImpl> spStreamImpl)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this stream on the device.
                        // NOTE: We have to store the plain pointer next to the weak pointer.
                        // This is necessary to find the entry on unregistering because the weak pointer will already be invalid at that point.
                        m_mapStreams.emplace(spStreamImpl.get(), spStreamImpl);
                    }
                    //-----------------------------------------------------------------------------
                    //! Unregisters the given stream from this device.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_HOST auto UnregisterStream(stream::cpu::detail::StreamCpuImpl const * const pStream) noexcept(false)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Unregister this stream from the device.
                        auto const itFind(std::find_if(
                            m_mapStreams.begin(),
                            m_mapStreams.end(),
                            [pStream](std::pair<stream::cpu::detail::StreamCpuImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuImpl>> const & pair)
                            {
                                return (pStream == pair.first);
                            }));
                        if(itFind != m_mapStreams.end())
                        {
                            m_mapStreams.erase(itFind);
                        }
                        else
                        {
                            throw std::logic_error("The stream to unregister from the device could not be found in the list of registered streams!");
                        }
                    }

                private:
                    std::mutex mutable m_Mutex;
                    std::map<stream::cpu::detail::StreamCpuImpl *, std::weak_ptr<stream::cpu::detail::StreamCpuImpl>> m_mapStreams;
                };
            }
        }

        //#############################################################################
        //! The CPU device handle.
        //#############################################################################
        class DevCpu
        {
            friend class DevManCpu;
        protected:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DevCpu() :
                m_spDevCpuImpl(std::make_shared<cpu::detail::DevCpuImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            //! Copy constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DevCpu(DevCpu const &) = default;
            //-----------------------------------------------------------------------------
            //! Move constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DevCpu(DevCpu &&) = default;
            //-----------------------------------------------------------------------------
            //! Copy assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(DevCpu const &) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            //! Move assignment operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator=(DevCpu &&) -> DevCpu & = default;
            //-----------------------------------------------------------------------------
            //! Destructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST ~DevCpu() = default;
            //-----------------------------------------------------------------------------
            //! Equality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator==(DevCpu const &) const
            -> bool
            {
                return true;
            }
            //-----------------------------------------------------------------------------
            //! Inequality comparison operator.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto operator!=(DevCpu const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }

        public:
            std::shared_ptr<cpu::detail::DevCpuImpl> m_spDevCpuImpl;
        };

        //#############################################################################
        //! The CPU device manager.
        //#############################################################################
        class DevManCpu
        {
        public:
            //-----------------------------------------------------------------------------
            //! Constructor.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST DevManCpu() = delete;

            //-----------------------------------------------------------------------------
            //! \return The number of devices available.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST static auto getDevCount()
            -> std::size_t
            {
                return 1;
            }
            //-----------------------------------------------------------------------------
            //! \return The number of devices available.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST static auto getDevByIdx(
                std::size_t const & uiIdx)
            -> DevCpu
            {
                std::size_t const uiNumDevices(getDevCount());
                if(uiIdx >= uiNumDevices)
                {
                    std::stringstream ssErr;
                    ssErr << "Unable to return device handle for device " << uiIdx << " because there are only " << uiNumDevices << " threads devices!";
                    throw std::runtime_error(ssErr.str());
                }

                return {};
            }
        };

        namespace cpu
        {
            //-----------------------------------------------------------------------------
            //! \return The device this object is bound to.
            //-----------------------------------------------------------------------------
            ALPAKA_FCT_HOST auto getDev()
            -> DevCpu
            {
                return DevManCpu::getDevByIdx(0);
            }
        }
    }
    
    namespace event
    {
        class EventCpuAsync;
    }
    namespace stream
    {
        class StreamCpuAsync;
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                dev::DevCpu>
            {
                using type = dev::DevCpu;
            };
            //#############################################################################
            //! The CPU device manager device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                dev::DevManCpu>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The CPU device device get trait specialization.
            //#############################################################################
            template<>
            struct GetDev<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getDev(
                    dev::DevCpu const & dev)
                -> dev::DevCpu
                {
                    return dev;
                }
            };

            //#############################################################################
            //! The CPU device name get trait specialization.
            //#############################################################################
            template<>
            struct GetName<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getName(
                    dev::DevCpu const & dev)
                -> std::string
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getCpuName();
                }
            };

            //#############################################################################
            //! The CPU device available memory get trait specialization.
            //#############################################################################
            template<>
            struct GetMemBytes<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    return dev::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device free memory get trait specialization.
            //#############################################################################
            template<>
            struct GetFreeMemBytes<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto getFreeMemBytes(
                    dev::DevCpu const & dev)
                -> std::size_t
                {
                    boost::ignore_unused(dev);

                    // \FIXME: Get correct free memory size!
                    return dev::cpu::detail::getGlobalMemSizeBytes();
                }
            };

            //#############################################################################
            //! The CPU device reset trait specialization.
            //#############################################################################
            template<>
            struct Reset<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto reset(
                    dev::DevCpu const & dev)
                -> void
                {
                    boost::ignore_unused(dev);

                    // The CPU does nothing on reset.
                }
            };

            //#############################################################################
            //! The CPU device device manager type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                dev::DevCpu>
            {
                using type = dev::DevManCpu;
            };
        }
    }
    namespace event
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                dev::DevCpu>
            {
                using type = event::EventCpuAsync;
            };
        }
    }
    namespace stream
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                dev::DevCpu>
            {
                using type = stream::StreamCpuAsync;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU device thread wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or streams that are created after this call is made are not waited for.
            //#############################################################################
            template<>
            struct CurrentThreadWaitFor<
                dev::DevCpu>
            {
                ALPAKA_FCT_HOST static auto currentThreadWaitFor(
                    dev::DevCpu const & dev)
                -> void
                {
                    // Get all the streams on the device at the time of invocation.
                    // All streams added afterwards are ignored.
                    auto vspStreams(
                        dev.m_spDevCpuImpl->GetAllStreams());

                    // Enqueue an event in every stream on the device.
                    // \TODO: This should be done atomically for all streams. 
                    // Furthermore there should not even be a chance to enqueue something between getting the streams and adding our wait events!
                    std::vector<event::EventCpuAsync> vEvents;
                    for(auto && spStream : vspStreams)
                    {
                        vEvents.emplace_back(dev);
                        stream::enqueue(spStream, vEvents.back());
                    }

                    // Now wait for all the events.
                    for(auto && event : vEvents)
                    {
                        wait::wait(event);
                    }
                }
            };
        }
    }
}