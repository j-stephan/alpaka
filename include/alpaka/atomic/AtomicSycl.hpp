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

#include <alpaka/core/Unused.hpp>
#include <alpaka/atomic/Op.hpp>
#include <alpaka/atomic/Traits.hpp>
#include <alpaka/meta/DependentFalseType.hpp>

#include <climits>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The SYCL accelerator atomic ops.
        //
        //  Atomics can used in the hierarchy level grids, blocks and threads.
        //  Atomics are not guaranteed to be save between devices
        class AtomicSycl
        {
        public:

            //-----------------------------------------------------------------------------
            AtomicSycl() = default;
            //-----------------------------------------------------------------------------
            AtomicSycl(AtomicSycl const &) = delete;
            //-----------------------------------------------------------------------------
            AtomicSycl(AtomicSycl &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicSycl const &) -> AtomicSycl & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AtomicSycl &&) -> AtomicSycl & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicSycl() = default;
        };

        namespace traits
        {
            //#############################################################################
            //! The specializations to execute the requested atomic ops of the SYCL accelerator.

            //-----------------------------------------------------------------------------
            // Add.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &, 
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    auto sycl_addr = cl::sycl::global_ptr<T>{addr};
                    auto atomic_addr = cl::sycl::atomic<T>{sycl_addr};

                    return cl::sycl::atomic_fetch_add(atomic_addr, value);
                }
            };

            //-----------------------------------------------------------------------------
            // Sub.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_sub(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Min.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
               typename T,
               typename THierarchy>
            struct AtomicOp<
                op::Min,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_min(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Max.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Max,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_max(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Exch.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_exchange(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Inc.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Inc,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // work-around with compare_and_exchange
                }
            };

            //-----------------------------------------------------------------------------
            // Dec.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Dec,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // work-around with compare_and_exchange
                }
            };

            //-----------------------------------------------------------------------------
            // And.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_and(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Or.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_or(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Xor.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_fetch_xor(atomic<T, addressSpace> object, T operand)
                }
            };

            //-----------------------------------------------------------------------------
            // Cas.

            //-----------------------------------------------------------------------------
            //! The SYCL accelerator atomic operation.
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Cas,
                atomic::AtomicSycl,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                static auto atomicOp(
                    atomic::AtomicSycl const &,
                    T * const addr,
                    T const & compare,
                    T const & value)
                -> T
                {
                    // TODO
                    // return cl::sycl::atomic_compare_exchange_strong(atomic<T, addressSpace object, T& expected, T desired)
                }
            };
        }
    }
}

#endif
