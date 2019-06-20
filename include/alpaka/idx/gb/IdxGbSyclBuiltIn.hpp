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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Sycl.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    namespace idx
    {
        namespace gb
        {
            //#############################################################################
            //! The SYCL accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxGbSyclBuiltIn
            {
            public:
                using IdxGbBase = IdxGbSyclBuiltIn;
                //-----------------------------------------------------------------------------
                IdxGbSyclBuiltIn() = default;
                //-----------------------------------------------------------------------------
                explicit IdxGbSyclBuiltIn(cl::sycl::nd_item<TDim::value> work_item)
                : my_item{work_item}
                {}
                //-----------------------------------------------------------------------------
                IdxGbSyclBuiltIn(IdxGbSyclBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxGbSyclBuiltIn(IdxGbSyclBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbSyclBuiltIn const & ) -> IdxGbSyclBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxGbSyclBuiltIn &&) -> IdxGbSyclBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxGbSyclBuiltIn() = default;

                cl::sycl::nd_item<TDim::value> my_item;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::gb::IdxGbSyclBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                idx::gb::IdxGbSyclBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    idx::gb::IdxGbSyclBuiltIn<TDim, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<TDim, TIdx>
                {
                    if constexpr(TDim::value == 1)
                    {
                        return vec::Vec<TDim, TIdx>(idx.my_item.get_group(0));
                    }
                    else if constexpr(TDim::Value == 2)
                    {
                        return vec::Vec<TDim, TIdx>(idx.my_item.get_group(0),
                                                    idx.my_item.get_group(1));
                    }
                    else
                    {
                        return vec::Vec<TDim, TIdx>(idx.my_item.get_group(0),
                                                    idx.my_item.get_group(1),
                                                    idx.my_item.get_group(2));
                    }
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator grid block index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::gb::IdxGbSyclBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
