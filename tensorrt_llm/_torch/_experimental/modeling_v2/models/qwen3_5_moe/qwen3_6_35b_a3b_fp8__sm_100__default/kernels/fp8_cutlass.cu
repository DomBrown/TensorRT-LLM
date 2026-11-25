// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Include CUTLASS before the FFI utilities expose Tensor in the global namespace.
// clang-format off
#include <flashinfer/gemm/gemm_groupwise_sm100.cuh>
#include <flashinfer/gemm/group_gemm_fp8_groupwise_sm100.cuh>
#include "tvm_ffi_utils.h"
// clang-format on

using E4M3 = cutlass::float_e4m3_t;
using BF16 = cutlass::bfloat16_t;

void dense(TensorView a, TensorView b, TensorView sa, TensorView sb, TensorView output, TensorView workspace)
{
    auto status = flashinfer::gemm::CutlassGroupwiseScaledGEMMSM100<1, 128, 128, false, 1, E4M3, BF16>(
        workspace.data_ptr(), workspace.size(0), static_cast<E4M3*>(a.data_ptr()), static_cast<E4M3*>(b.data_ptr()),
        static_cast<float*>(sa.data_ptr()), static_cast<float*>(sb.data_ptr()), static_cast<BF16*>(output.data_ptr()),
        a.size(0), b.size(0), a.size(1), 1, get_stream(a.device()));
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

void grouped(TensorView a, TensorView b, TensorView sa, TensorView sb, TensorView output, TensorView extent,
    TensorView descriptors, TensorView workspace)
{
    auto status = flashinfer::group_gemm::CutlassFP8GroupwiseScaledGroupGEMMSM100<1, 128, 128, false, 1, E4M3, BF16>(
        descriptors.data_ptr(), descriptors.size(0), workspace.data_ptr(), workspace.size(0),
        static_cast<E4M3*>(a.data_ptr()), static_cast<E4M3*>(b.data_ptr()), static_cast<float*>(sa.data_ptr()),
        static_cast<float*>(sb.data_ptr()), static_cast<BF16*>(output.data_ptr()), static_cast<int*>(extent.data_ptr()),
        sa.size(1), b.size(1), b.size(2), b.size(0), get_stream(a.device()));
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dense, dense);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped, grouped);

void grouped_small(TensorView a, TensorView b, TensorView sa, TensorView sb, TensorView output, TensorView extent,
    TensorView descriptors, TensorView workspace)
{
    using namespace cute;
    using namespace cutlass;
    using Tile = Shape<_64, _256, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<1, 128, 128, UMMA::Major::MN, UMMA::Major::MN>;
    using ScaleA = decltype(ScaleConfig::deduce_layoutSFA());
    using ScaleB = decltype(ScaleConfig::deduce_layoutSFB());

    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<arch::Sm100, arch::OpClassTensorOp, Tile,
        Cluster, epilogue::collective::EpilogueTileAuto, float, float, void, void*, 0, BF16, layout::RowMajor*, 8,
        epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
    using Mainloop = typename gemm::collective::CollectiveBuilder<arch::Sm100, arch::OpClassTensorOp, E4M3,
        tuple<layout::RowMajor*, ScaleA*>, 16, E4M3, tuple<layout::ColumnMajor*, ScaleB*>, 16, float, Tile, Cluster,
        gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        gemm::KernelPtrArrayTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;
    using Groups = gemm::GroupProblemShape<Shape<int, int, int>>;
    using Problem = Groups::UnderlyingProblemShape;
    using Kernel = gemm::kernel::GemmUniversal<Groups, Mainloop, Epilogue, void>;
    using Gemm = gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = Kernel::InternalStrideA;
    using StrideB = Kernel::InternalStrideB;
    using StrideD = Kernel::InternalStrideD;

    int experts = b.size(0);
    flashinfer::AlignedAllocator metadata(descriptors.data_ptr(), descriptors.size(0));
    auto shapes = metadata.aligned_alloc<Problem>(experts * sizeof(Problem), 16, "shapes");
    auto act = metadata.aligned_alloc<const E4M3*>(experts * sizeof(E4M3*), 16, "A");
    auto weights = metadata.aligned_alloc<const E4M3*>(experts * sizeof(E4M3*), 16, "B");
    auto out = metadata.aligned_alloc<BF16*>(experts * sizeof(BF16*), 16, "D");
    auto act_scale = metadata.aligned_alloc<float const*>(experts * sizeof(float*), 16, "SA");
    auto weight_scale = metadata.aligned_alloc<float const*>(experts * sizeof(float*), 16, "SB");
    auto stride_a = metadata.aligned_alloc<StrideA>(experts * sizeof(StrideA), 16, "stride A");
    auto stride_b = metadata.aligned_alloc<StrideB>(experts * sizeof(StrideB), 16, "stride B");
    auto stride_d = metadata.aligned_alloc<StrideD>(experts * sizeof(StrideD), 16, "stride D");
    auto layout_a = metadata.aligned_alloc<ScaleA>(experts * sizeof(ScaleA), 16, "layout SA");
    auto layout_b = metadata.aligned_alloc<ScaleB>(experts * sizeof(ScaleB), 16, "layout SB");

    cudaStream_t stream = get_stream(a.device());
    cudaLaunchAttribute pdl{};
    pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl.val.programmaticStreamSerializationAllowed = true;
    cudaLaunchConfig_t launch{};
    launch.gridDim = (experts + 255) / 256;
    launch.blockDim = 256;
    launch.stream = stream;
    launch.attrs = &pdl;
    launch.numAttrs = 1;

    auto setup = flashinfer::group_gemm::compute_sm100_cutlass_group_gemm_args<ScaleConfig, E4M3, float, BF16, Problem,
        StrideA, StrideB, StrideD, ScaleA, ScaleB, false>;
    auto status = cudaLaunchKernelEx(&launch, setup, static_cast<E4M3*>(a.data_ptr()), static_cast<E4M3*>(b.data_ptr()),
        static_cast<float*>(sa.data_ptr()), static_cast<float*>(sb.data_ptr()), static_cast<BF16*>(output.data_ptr()),
        static_cast<int*>(extent.data_ptr()), static_cast<int>(sa.size(1)), static_cast<int>(b.size(1)),
        static_cast<int>(b.size(2)), experts, 1, 128, 128, shapes, act, weights, act_scale, weight_scale, out, stride_a,
        stride_b, stride_d, layout_a, layout_b);
    TVM_FFI_ICHECK(status == cudaSuccess) << cudaGetErrorString(status);

    thread_local int const sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();
    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = 0;
    hardware.sm_count = sm_count;
    typename Gemm::Arguments args{gemm::GemmUniversalMode::kGrouped, {experts, shapes, nullptr},
        {act, stride_a, weights, stride_b, act_scale, layout_a, weight_scale, layout_b},
        {{}, nullptr, nullptr, out, stride_d}, hardware};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;

    Gemm kernel;
    TVM_FFI_ICHECK(Gemm::get_workspace_size(args) <= workspace.size(0));
    TVM_FFI_ICHECK(kernel.can_implement(args) == cutlass::Status::kSuccess);
    TVM_FFI_ICHECK(kernel.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess);
    TVM_FFI_ICHECK(kernel.run(stream, nullptr, true) == cutlass::Status::kSuccess);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(grouped_small, grouped_small);
