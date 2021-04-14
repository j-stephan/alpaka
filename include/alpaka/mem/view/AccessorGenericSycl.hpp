/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#include <alpaka/mem/view/Accessor.hpp>

#include <sycl/sycl.hpp>

#include <utility>
#include <tuple>
#include <type_traits>

namespace alpaka
{
    namespace detail
    {
        template <typename TAccessMode>
        struct SYCLMode
        {
        };

        template <>
        struct SYCLMode<ReadAccess>
        {
            static constexpr auto value = sycl::access::mode::read;
        };

        template <>
        struct SYCLMode<WriteAccess>
        {
            static constexpr auto value = sycl::access::mode::write;
        };

        template <>
        struct SYCLMode<ReadWriteAccess>
        {
            static constexpr auto value = sycl::access::mode::read_write;
        };


        template <typename... TAccessModes>
        struct SYCLMode<std::tuple<TAccessModes...>>
        {
            static constexpr auto value = sycl::access::mode::read_write;
        };
    }

    template <typename TElem, typename TIdx, typename TAccessModes, sycl::access::mode TSYCLMode, typename TProperties>
    struct Accessor<sycl::accessor<TElem, 1, TSYCLMode, sycl::access::target::global_buffer,
                                   sycl::access::placeholder::true_t, TProperties>,
                    TElem, TIdx, std::size_t{1}, TAccessModes>
    {
        using HandleType = sycl::accessor<TElem, 1, TSYCLMode, sycl::access::target::global_buffer,
                                          sycl::access::placeholder::true_t, TProperties>;
        using ReturnType = std::conditional_t<(TSYCLMode == sycl::access::mode::read), typename HandleType::value_type, typename HandleType::reference>;

        Accessor(HandleType acc, Vec<DimInt<1>, TIdx> ext) : m_acc{acc}, extents{ext}
        {
        }

        Accessor(Accessor const& other) = default;
        auto operator=(Accessor const& rhs) -> Accessor& = default;

        template <typename TOtherAccessModes>
        Accessor(Accessor<HandleType, TElem, TIdx, std::size_t{1}, TOtherAccessModes> const& other)
        : m_acc{other.m_acc} // This will fail if HandleType != other::HandleType
        , extents{other.extents}
        {
        }

        ~Accessor() = default;

        auto operator[](Vec<DimInt<1>, std::size_t> i) const -> ReturnType
        {
            return operator()(i[0]);
        }

        auto operator[](std::size_t i) const -> ReturnType
        {
            return operator()(i);
        }

        auto operator()(std::size_t x) const -> ReturnType
        {
            auto const id = sycl::id<1>{x};
            return m_acc[id];
        }

        HandleType m_acc;
        Vec<DimInt<1>, TIdx> extents;
    };

    template <typename TElem, typename TIdx, typename TAccessModes, sycl::access::mode TSYCLMode, typename TProperties>
    struct Accessor<sycl::accessor<TElem, 2, TSYCLMode, sycl::access::target::global_buffer,
                                   sycl::access::placeholder::true_t, TProperties>,
                    TElem, TIdx, std::size_t{2}, TAccessModes>
    {
        using HandleType = sycl::accessor<TElem, 2, TSYCLMode, sycl::access::target::global_buffer,
                                          sycl::access::placeholder::true_t, TProperties>;
        using ReturnType = std::conditional_t<(TSYCLMode == sycl::access::mode::read), typename HandleType::value_type, typename HandleType::reference>;

        Accessor(HandleType acc, Vec<DimInt<2>, TIdx> ext) : m_acc{acc}, extents{ext}
        {
        }

        Accessor(Accessor const& other) = default;
        auto operator=(Accessor const& rhs) -> Accessor& = default;

        auto operator=(Accessor&& rhs) -> Accessor&
        {
            std::swap(m_acc, rhs.m_acc);
            std::swap(extents, rhs.extents);
            return *this;
        }

        ~Accessor() = default;

        auto operator[](Vec<DimInt<2>, std::size_t> i) const -> ReturnType
        {
            return operator()(i[0], i[1]);
        }

        auto operator()(std::size_t y, std::size_t x) const -> ReturnType
        {
            auto const id = sycl::id<2>{y, x};
            return m_acc[id];
        }

        HandleType m_acc;
        Vec<DimInt<2>, TIdx> extents;
    };

    template <typename TElem, typename TIdx, typename TAccessModes, sycl::access::mode TSYCLMode, typename TProperties>
    struct Accessor<sycl::accessor<TElem, 3, TSYCLMode, sycl::access::target::global_buffer,
                                   sycl::access::placeholder::true_t, TProperties>,
                    TElem, TIdx, std::size_t{3}, TAccessModes>
    {
        using HandleType = sycl::accessor<TElem, 3, TSYCLMode, sycl::access::target::global_buffer,
                                          sycl::access::placeholder::true_t>;
        using ReturnType = std::conditional_t<(TSYCLMode == sycl::access::mode::read), typename HandleType::value_type, typename HandleType::reference>;

        Accessor(HandleType acc, Vec<DimInt<3>, TIdx> ext) : m_acc{acc}, extents{ext}
        {
        }

        Accessor(Accessor const& other) = default;
        auto operator=(Accessor const& rhs) -> Accessor& = default;

        template <typename TOtherAccessModes>
        Accessor(Accessor<HandleType, TElem, TIdx, std::size_t{3}, TOtherAccessModes> const& other)
        : m_acc{other.m_acc} // This will fail if HandleType != other::HandleType
        , extents{other.extents}
        {
        }

        ~Accessor() = default;

        auto operator[](Vec<DimInt<3>, std::size_t> i) const -> ReturnType
        {
            return operator()(i[0], i[1], i[2]);
        }

        auto operator()(std::size_t z, std::size_t y, std::size_t x) const -> ReturnType
        {
            auto const id = sycl::id<3>{z, y, x};
            return m_acc[id];
        }

        HandleType m_acc;
        Vec<DimInt<3>, TIdx> extents;
    };
}

#endif
