/* Copyright 2021 Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <alpaka/example/ExampleDefaultAcc.hpp>
#include <alpaka/mem/view/Accessor.hpp>

#include <catch2/catch.hpp>

namespace
{
    constexpr auto N = 1024;

    struct WriteKernelTemplate
    {
        template<typename Acc, typename Accessor>
        ALPAKA_FN_ACC void operator()(Acc const& acc, Accessor data) const
        {
            (void) acc;
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::Dim<Acc>, alpaka::Idx<Acc>>{alpaka::Idx<Acc>{3}}] = 3.0f;
        }
    };

    struct WriteKernelExplicit
    {
        template<typename Acc, typename Pointer, typename Idx>
        ALPAKA_FN_ACC void operator()(Acc const& acc, alpaka::Accessor<Pointer, float, Idx, 1> const data) const
        {
            (void) acc;
            data[1] = 1.0f;
            data(2) = 2.0f;
            data[alpaka::Vec<alpaka::Dim<Acc>, Idx>{Idx{3}}] = 3.0f;
        }
    };

    struct ReadKernelTemplate
    {
        template<typename Acc, typename Accessor>
        ALPAKA_FN_ACC void operator()(Acc const& acc, Accessor data) const
        {
            (void) acc;
            const float v1 = data[1];
            const float v2 = data(2);
            const float v3 = data[alpaka::Vec<alpaka::Dim<Acc>, alpaka::Idx<Acc>>{alpaka::Idx<Acc>{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };

    struct ReadKernelExplicit
    {
        template<typename Acc, typename Pointer, typename Idx>
        ALPAKA_FN_ACC void operator()(Acc const& acc, alpaka::Accessor<Pointer, const float, Idx, 1> const data) const
        {
            (void) acc;
            const float v1 = data[1];
            const float v2 = data(2);
            const float v3 = data[alpaka::Vec<alpaka::Dim<Acc>, Idx>{Idx{3}}];
            (void) v1;
            (void) v2;
            (void) v3;
        }
    };
} // namespace

TEST_CASE("readWrite", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<float, Size>(devAcc, Size{N});
    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};

    alpaka::exec<Acc>(queue, workdiv, WriteKernelTemplate{}, alpaka::access(buffer));
    alpaka::exec<Acc>(queue, workdiv, WriteKernelExplicit{}, alpaka::access(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelTemplate{}, alpaka::readAccess(buffer));
    alpaka::exec<Acc>(queue, workdiv, ReadKernelExplicit{}, alpaka::readAccess(buffer));
}

namespace
{
    struct MyPointer
    {
        ALPAKA_FN_ACC auto operator[](std::size_t i) const -> float&
        {
            return p[i];
        }

        float* p;
    };
} // namespace

TEST_CASE("customPointer", "[accessor]")
{
    using Dim = alpaka::DimInt<1>;
    using Size = std::size_t;
    using Acc = alpaka::ExampleDefaultAcc<Dim, Size>;
    using DevAcc = alpaka::Dev<Acc>;
    using Queue = alpaka::Queue<DevAcc, alpaka::Blocking>;

    auto const devAcc = alpaka::getDevByIdx<Acc>(0u);
    auto queue = Queue{devAcc};
    auto buffer = alpaka::allocBuf<float, Size>(devAcc, Size{N});
    const auto workdiv = alpaka::WorkDivMembers<Dim, Size>{
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}},
        alpaka::Vec<Dim, Size>{Size{1}}};

    // TODO: Accessor is invoking UB here by reinterpreting as MyPointer ...
    auto accessor = alpaka::Accessor<MyPointer, float, alpaka::Idx<Acc>, Dim::value>{
        {alpaka::getPtrNative(buffer)},
        {alpaka::extent::getExtent<0>(buffer)}};
    auto readAccessor = alpaka::Accessor<const MyPointer, const float, alpaka::Idx<Acc>, Dim::value>{
        {alpaka::getPtrNative(buffer)},
        {alpaka::extent::getExtent<0>(buffer)}};

    alpaka::exec<Acc>(queue, workdiv, WriteKernelTemplate{}, accessor);
    alpaka::exec<Acc>(queue, workdiv, WriteKernelExplicit{}, accessor);
    alpaka::exec<Acc>(queue, workdiv, ReadKernelTemplate{}, readAccessor);
    alpaka::exec<Acc>(queue, workdiv, ReadKernelExplicit{}, readAccessor);
}
