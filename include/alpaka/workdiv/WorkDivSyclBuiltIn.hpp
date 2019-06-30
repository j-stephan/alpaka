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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Sycl.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The SYCL accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivSyclBuiltIn
        {
        public:
            using WorkDivBase = WorkDivSyclBuiltIn;

            //-----------------------------------------------------------------------------
            WorkDivSyclBuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                cl::sycl::nd_item<TDim::value> work_item)
                : m_threadElemExtent{threadElemExtent}, my_item{work_item}
            {}
            //-----------------------------------------------------------------------------
            WorkDivSyclBuiltIn(WorkDivSyclBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            WorkDivSyclBuiltIn(WorkDivSyclBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivSyclBuiltIn const &) -> WorkDivSyclBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivSyclBuiltIn &&) -> WorkDivSyclBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivSyclBuiltIn() = default;

        public:
            vec::Vec<TDim, TIdx> const & m_threadElemExtent;
            cl::sycl::nd_item<TDim::value> my_item;

        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivSyclBuiltIn<TDim, TIdx>>
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
            //! The SYCL accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivSyclBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The SYCL accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivSyclBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivSyclBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    if constexpr(TDim::value == 1)
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_group_range(0)};
                    }
                    else if constexpr(TDim::value == 2)
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_group_range(0),
                                                    workDiv.my_item.get_group_range(1)};
                    }
                    else
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_group_range(0),
                                                    workDiv.my_item.get_group_range(1),
                                                    workDiv.my_item.get_group_range(2)};
                    }
                }
            };

            //#############################################################################
            //! The SYCL accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivSyclBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                static auto getWorkDiv(
                    WorkDivSyclBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    if constexpr(TDim::value == 1)
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_local_range(0)};
                    }
                    else if constexpr(TDim::value == 2)
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_local_range(0),
                                                    workDiv.my_item.get_local_range(1)};
                    }
                    else
                    {
                        return vec::Vec<TDim, TIdx>{workDiv.my_item.get_local_range(0),
                                                    workDiv.my_item.get_local_range(1),
                                                    workDiv.my_item.get_local_range(2)};
                    }
                }
            };

            //#############################################################################
            //! The SYCL accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivSyclBuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivSyclBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
