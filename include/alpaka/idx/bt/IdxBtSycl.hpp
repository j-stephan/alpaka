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
    namespace bt
    {
        //#############################################################################
        //! The SYCL accelerator ND index provider.
        template<typename TDim, typename TIdx>
        class IdxBtSycl : public concepts::Implements<ConceptIdxBt, IdxBtSycl<TDim, TIdx>>
        {
        public:
            using IdxBtBase = IdxBtSycl;

            //-----------------------------------------------------------------------------
            IdxBtSycl() = default;
            //-----------------------------------------------------------------------------
            explicit IdxBtSycl(cl::sycl::nd_item<TDim::value> work_item)
            : my_item{work_item} {}
            //-----------------------------------------------------------------------------
            IdxBtSycl(IdxBtSycl const &) = default;
            //-----------------------------------------------------------------------------
            IdxBtSycl(IdxBtSycl &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtSycl const & ) -> IdxBtSycl & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtSycl &&) -> IdxBtSycl & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~IdxBtSycl() = default;

            cl::sycl::nd_item<TDim::value> my_item;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The SYCL accelerator index dimension get trait specialization.
        template<typename TDim, typename TIdx>
        struct DimType<bt::IdxBtSycl<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The SYCL accelerator block thread index get trait specialization.
        template<typename TDim, typename TIdx>
        struct GetIdx<bt::IdxBtSycl<TDim, TIdx>, origin::Block, unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<typename TWorkDiv>
            static auto getIdx(bt::IdxBtSycl<TDim, TIdx> const & idx, TWorkDiv const &) -> Vec<TDim, TIdx>
            {
                if constexpr(TDim::value == 1)
                    return Vec<TDim, TIdx>{idx.my_item.get_local_id(0)};
                else if constexpr(TDim::value == 2)
                    return Vec<TDim, TIdx>{idx.my_item.get_local_id(0), idx.my_item.get_local_id(1)};
                else
                {
                    return Vec<TDim, TIdx>{idx.my_item.get_local_id(0), idx.my_item.get_local_id(1),
                                           idx.my_item.get_local_id(2)};
                }
            }
        };

        //#############################################################################
        //! The SYCL accelerator block thread index idx type trait specialization.
        template<typename TDim, typename TIdx>
        struct IdxType<bt::IdxBtSycl<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
