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
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/mem/buf/sycl/Utility.hpp>
#include <alpaka/queue/Traits.hpp>

#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Sycl.hpp>

namespace alpaka
{
    namespace dev
    {
        class DevSycl;
    }
}

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace sycl
            {
                namespace detail
                {
                    //#############################################################################
                    //! The SYCL memory set trait.
                    template<
                        typename TElem,
                        typename TDim>
                    struct TaskSetSycl
                    {
                        //-----------------------------------------------------------------------------
                        auto operator()(cl::sycl::handler& cgh) -> void
                        {
                            auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh, range);
                            cgh.fill(acc, value);
                        }

                        cl::sycl::buffer<TElem, dim::Dim<TDim>::value>& buf;
                        TElem value;
                        cl::sycl::range<dim::Dim<TDim>::value> range;
                    };
                }
            }

            namespace traits
            {
                //#############################################################################
                //! The SYCL device memory set trait specialization.
                template<
                    typename TDim>
                struct CreateTaskSet<
                    TDim,
                    dev::DevSycl>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TView>
                    ALPAKA_FN_HOST static auto createTaskSet(
                        TView & view,
                        std::uint8_t const & byte,
                        TExtent const & extent)
                    {

                        // SYCL doesn't support byte-wise filling. So we create
                        // a corresponding element of our actual type which
                        // is a concatenation of the correct number of bytes.
                        using value_type = elem::Elem<TView>;

                        // First step: Create data type of same size and
                        // alignment as the actual data type. Our data type
                        // might not be trivially constructible so we have to
                        // do this the hard way.
                        using data_type = std::aligned_storage_t<
                                            sizeof(value_type),
                                            alignof(value_type)>;

                        // Second step: Create single element and initialize
                        // to our byte value.
                        auto data_value = data_type{};
                        std::memset(&data_value, byte, sizeof(data_type));

                        // Third step: Reinterpret element as actual data
                        // type. std::launder is a requirement of C++17
                        auto element = *std::launder(reinterpret_cast<value_type*>(&data_value));
                        
                        // Fourth step: Create task. We pass the element by
                        // value so it gets copied. This is required because the
                        // uninitialized storage is allocated with placement new
                        // and we have to call the destructor by hand to prevent
                        // memory leaks.
                        auto task = mem::view::sycl::detail::TaskSetSycl<
                                        value_type, 
                                        TDim>{
                                            view.m_buf,
                                            element,
                                            mem::view::sycl::detail::get_sycl_range<dim::Dim<TExtent>::value>(extent)
                                        };

                        // Destroy dummy object
                        std::launder(reinterpret_cast<value_type*>(&data_value))->~value_type();

                        // Return task
                        return task;
                    }
                };
            }
        }
    }
}

#endif
