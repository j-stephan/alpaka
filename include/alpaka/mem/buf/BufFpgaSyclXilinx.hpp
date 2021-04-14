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
#include <alpaka/mem/buf/BufGenericSycl.hpp>

namespace alpaka
{
    
    //#############################################################################
    //! The SYCL memory buffer.
    template<typename TElem, typename TDim, typename TIdx>
    using BufFpgaSyclXilinx = BufGenericSycl<TElem, TDim, TIdx, DevFpgaSyclXilinx>;

    namespace traits
    {
        //! The customization point for how to build an accessor for a given memory object.
        template<typename TElem, typename TDim, typename TIdx>
        struct BuildAccessor<BufFpgaSyclXilinx<TElem, TDim, TIdx>>
        {
            template<typename... TAccessModes>
            ALPAKA_FN_HOST_ACC static auto buildAccessor(BufFpgaSyclXilinx<TElem, TDim, TIdx> const& buffer)
            {
                constexpr auto SYCLMode = alpaka::detail::SYCLMode<TAccessModes...>::value;
                using Modes = typename traits::internal::BuildAccessModeList<TAccessModes...>::type;

                auto buf = buffer.m_buf; // buffers are reference counted, so we can copy to work around constness

// bug in compiler: https://github.com/intel/llvm/issues/3536
// TODO Uncomment once fixed!
// #if defined(ALPAKA_SYCL_BACKEND_XILINX)
#if 0
                // Prevent v++ from placing everything in DDR bank 0.
                constexpr auto bank_counter = __COUNTER__ % ALPAKA_SYCL_XILINX_DDR_BANKS; // user-defined (CMake)
                using bank_assignment = sycl::xilinx::property::ddr_bank::instance<bank_counter>;
                using PL = sycl::ONEAPI::accessor_property_list<bank_assignment>;

                using SYCLAcc = sycl::accessor<TElem, int{TDim::value}, SYCLMode, sycl::access::target::global_buffer,
                                               sycl::access::placeholder::true_t, PL>;

                auto const properties = PL{sycl::xilinx::ddr_bank<bank_counter>};
                auto sycl_acc = SYCLAcc{buf, properties};
                using Acc = Accessor<SYCLAcc, TElem, TIdx, std::size_t{TDim::value}, Modes>;
                return Acc{sycl_acc, buffer.m_extentElements};
#else
                using SYCLAcc = sycl::accessor<TElem, int{TDim::value}, SYCLMode, sycl::access::target::global_buffer,
                                               sycl::access::placeholder::true_t>;
                using Acc = Accessor<SYCLAcc, TElem, TIdx, std::size_t{TDim::value}, Modes>;

                return Acc{SYCLAcc{buf}, buffer.m_extentElements};
#endif
            }
        };
    }
}

#endif
