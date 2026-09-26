

import torch
from torch._inductor.codecache import CppWrapperCodeCache

cpp_wrapper_src = (
r"""
#include <torch/csrc/inductor/cpp_wrapper/cpu.h>
extern "C"  void  cpp_fused__to_copy_add_bitwise_and_bitwise_xor_ne_sum_view_0(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       int32_t* out_ptr0,
                       int64_t* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1);
extern "C"  void  cpp_fused_cat_index_new_zeros_slice_sub_1(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const int64_t* in_ptr3,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       int64_t* out_ptr3,
                       int64_t* out_ptr4,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2);
CACHE_TORCH_DTYPE(int32);
CACHE_TORCH_DTYPE(int64);
CACHE_TORCH_DEVICE(cpu);

void inductor_entry_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles  // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed)
) {
    py::gil_scoped_release_simple release;
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 7);
    int64_t arg0_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[0], &arg0_1));
    auto arg1_1 = std::move(inputs[1]);
    int64_t arg2_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[2], &arg2_1));
    int64_t arg3_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[3], &arg3_1));
    auto arg4_1 = std::move(inputs[4]);
    int64_t arg5_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[5], &arg5_1));
    auto arg6_1 = std::move(inputs[6]);
    int64_t s65 = arg0_1;
    int64_t s37 = arg2_1;
    int64_t s11 = arg3_1;
    int64_t s47 = arg5_1;
    assert_size_stride(arg1_1, {s65, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [view], Original ATen: [aten.view]
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg1_1, cached_torch_dtype_int32, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    assert_size_stride(buf1, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
    assert_size_stride(arg4_1, {s11, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [view_2], Original ATen: [aten.view]
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg4_1, cached_torch_dtype_int32, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    assert_size_stride(buf3, {c10::div_floor_integer(static_cast<int64_t>(s11), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
    const int64_t int_array_0[] = {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(s37), static_cast<int64_t>(4L))))), };
    static constexpr int64_t int_array_1[] = {1L, };
    AtenTensorHandle buf4_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_int32, cached_torch_device_type_cpu, 0, &buf4_handle));
    RAIIAtenTensorHandle buf4(buf4_handle);
    AtenTensorHandle buf5_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_int64, cached_torch_device_type_cpu, 0, &buf5_handle));
    RAIIAtenTensorHandle buf5(buf5_handle);
    cpp_fused__to_copy_add_bitwise_and_bitwise_xor_ne_sum_view_0((const int32_t*)(buf1.data_ptr()), (const int32_t*)(buf3.data_ptr()), (int32_t*)(buf4.data_ptr()), (int64_t*)(buf5.data_ptr()), s37, s65);
    arg1_1.reset();
    arg4_1.reset();

    buf1.reset();

    buf3.reset();
    buf4.reset();
    // Topologically Sorted Source Nodes: [counts_4, cumsum], Original ATen: [aten._to_copy, aten.cumsum]
    int32_t var_0 = cached_torch_dtype_int64;
    AtenTensorHandle buf7_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_cumsum(buf5, 0L, &var_0, &buf7_handle));
    RAIIAtenTensorHandle buf7(buf7_handle);
    buf5.reset();
    assert_size_stride(buf7, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(s37), static_cast<int64_t>(4L))))), }, {1L, }, "torch.ops.aten.cumsum.default");

    const int64_t int_array_2[] = {1L + (c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(s37), static_cast<int64_t>(4L)))))), };
    AtenTensorHandle buf10_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_2, int_array_1, cached_torch_dtype_int64, cached_torch_device_type_cpu, 0, &buf10_handle));
    RAIIAtenTensorHandle buf10(buf10_handle);
    static constexpr int64_t int_array_3[] = {1L, };
    auto buf8 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf10, 1, int_array_3, int_array_3, 0L));  // alias
    const int64_t int_array_4[] = {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(s37), static_cast<int64_t>(4L))))), };
    auto buf9 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf10, 1, int_array_4, int_array_3, 1L));  // alias
    const int64_t int_array_5[] = {s47, };
    AtenTensorHandle buf13_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_5, int_array_1, cached_torch_dtype_int64, cached_torch_device_type_cpu, 0, &buf13_handle));
    RAIIAtenTensorHandle buf13(buf13_handle);
    auto buf11 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf13, 1, int_array_3, int_array_3, 0L));  // alias
    assert_size_stride(arg6_1, {s47, }, {1L, }, "input");
    const int64_t int_array_6[] = {(-1L) + s47, };
    auto buf12 = wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(buf13, 1, int_array_6, int_array_3, 1L));  // alias
    AtenTensorHandle buf14_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_5, int_array_1, cached_torch_dtype_int64, cached_torch_device_type_cpu, 0, &buf14_handle));
    RAIIAtenTensorHandle buf14(buf14_handle);
    cpp_fused_cat_index_new_zeros_slice_sub_1((const int64_t*)(buf7.data_ptr()), (const int64_t*)(arg6_1.data_ptr()), (const int64_t*)(buf10.data_ptr()), (const int64_t*)(buf13.data_ptr()), (int64_t*)(buf8.data_ptr()), (int64_t*)(buf9.data_ptr()), (int64_t*)(buf11.data_ptr()), (int64_t*)(buf12.data_ptr()), (int64_t*)(buf14.data_ptr()), s37, s65, s47);
    arg6_1.reset();
    output_handles[0] = buf14.release();
} // inductor_entry_impl

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  cpp_fused__to_copy_add_bitwise_and_bitwise_xor_ne_sum_view_0(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       int32_t* out_ptr0,
                       int64_t* out_ptr1,
                       const int64_t ks0,
                       const int64_t ks1)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))); x0+=static_cast<int64_t>(1L))
        {
            {
                int32_t tmp_acc0 = 0;
                at::vec::Vectorized<int32_t> tmp_acc0_vec = at::vec::Vectorized<int32_t>(0);
                for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))); x1+=static_cast<int64_t>(16L))
                {
                    {
                        if(C10_LIKELY(x1 >= static_cast<int64_t>(0) && x1 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))))))
                        {
                            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr0 + static_cast<int64_t>(x1 + x0*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))), static_cast<int64_t>(16));
                            auto tmp1 = at::vec::Vectorized<int32_t>::loadu(in_ptr1 + static_cast<int64_t>(x1 + x0*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))), static_cast<int64_t>(16));
                            auto tmp2 = tmp0 ^ tmp1;
                            auto tmp3 = static_cast<int32_t>(255);
                            auto tmp4 = at::vec::Vectorized<int32_t>(tmp3);
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = static_cast<int32_t>(0);
                            auto tmp7 = at::vec::Vectorized<int32_t>(tmp6);
                            auto tmp8 = at::vec::VecMask<int32_t,1>(tmp5 != tmp7);
                            auto tmp9 = tmp8.to<int32_t,1>();
                            auto tmp10 = static_cast<int32_t>(65280);
                            auto tmp11 = at::vec::Vectorized<int32_t>(tmp10);
                            auto tmp12 = tmp2 & tmp11;
                            auto tmp13 = at::vec::VecMask<int32_t,1>(tmp12 != tmp7);
                            auto tmp14 = tmp13.to<int32_t,1>();
                            auto tmp15 = tmp9 + tmp14;
                            auto tmp16 = static_cast<int32_t>(16711680);
                            auto tmp17 = at::vec::Vectorized<int32_t>(tmp16);
                            auto tmp18 = tmp2 & tmp17;
                            auto tmp19 = at::vec::VecMask<int32_t,1>(tmp18 != tmp7);
                            auto tmp20 = tmp19.to<int32_t,1>();
                            auto tmp21 = tmp15 + tmp20;
                            auto tmp22 = static_cast<int32_t>(-16777216);
                            auto tmp23 = at::vec::Vectorized<int32_t>(tmp22);
                            auto tmp24 = tmp2 & tmp23;
                            auto tmp25 = at::vec::VecMask<int32_t,1>(tmp24 != tmp7);
                            auto tmp26 = tmp25.to<int32_t,1>();
                            auto tmp27 = tmp21 + tmp26;
                            tmp_acc0_vec = tmp_acc0_vec + tmp27;
                        }
                        if(C10_UNLIKELY(x1 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L)))) && x1 < static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))
                        {
                            auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr0 + static_cast<int64_t>(x1 + x0*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                            auto tmp1 = at::vec::Vectorized<int32_t>::loadu(in_ptr1 + static_cast<int64_t>(x1 + x0*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                            auto tmp2 = tmp0 ^ tmp1;
                            auto tmp3 = static_cast<int32_t>(255);
                            auto tmp4 = at::vec::Vectorized<int32_t>(tmp3);
                            auto tmp5 = tmp2 & tmp4;
                            auto tmp6 = static_cast<int32_t>(0);
                            auto tmp7 = at::vec::Vectorized<int32_t>(tmp6);
                            auto tmp8 = at::vec::VecMask<int32_t,1>(tmp5 != tmp7);
                            auto tmp9 = tmp8.to<int32_t,1>();
                            auto tmp10 = static_cast<int32_t>(65280);
                            auto tmp11 = at::vec::Vectorized<int32_t>(tmp10);
                            auto tmp12 = tmp2 & tmp11;
                            auto tmp13 = at::vec::VecMask<int32_t,1>(tmp12 != tmp7);
                            auto tmp14 = tmp13.to<int32_t,1>();
                            auto tmp15 = tmp9 + tmp14;
                            auto tmp16 = static_cast<int32_t>(16711680);
                            auto tmp17 = at::vec::Vectorized<int32_t>(tmp16);
                            auto tmp18 = tmp2 & tmp17;
                            auto tmp19 = at::vec::VecMask<int32_t,1>(tmp18 != tmp7);
                            auto tmp20 = tmp19.to<int32_t,1>();
                            auto tmp21 = tmp15 + tmp20;
                            auto tmp22 = static_cast<int32_t>(-16777216);
                            auto tmp23 = at::vec::Vectorized<int32_t>(tmp22);
                            auto tmp24 = tmp2 & tmp23;
                            auto tmp25 = at::vec::VecMask<int32_t,1>(tmp24 != tmp7);
                            auto tmp26 = tmp25.to<int32_t,1>();
                            auto tmp27 = tmp21 + tmp26;
                            tmp_acc0_vec = sum_masked_reduce(tmp_acc0_vec, tmp27, static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                        }
                    }
                }
                tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<int32_t, 1>([](at::vec::Vectorized<int32_t>& x, at::vec::Vectorized<int32_t>& y) { return x + y; }, tmp_acc0_vec);
                out_ptr0[static_cast<int64_t>(x0)] = static_cast<int32_t>(tmp_acc0);
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))
                {
                    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = at::vec::convert<int64_t,2,int32_t,1>(tmp0);
                    tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))) && x0 < static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))))
                {
                    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))) + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))));
                    auto tmp1 = at::vec::convert<int64_t,2,int32_t,1>(tmp0);
                    tmp1.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))) + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))));
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  cpp_fused_cat_index_new_zeros_slice_sub_1(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       const int64_t* in_ptr3,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       int64_t* out_ptr3,
                       int64_t* out_ptr4,
                       const int64_t ks0,
                       const int64_t ks1,
                       const int64_t ks2)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr0[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))) && x0 < static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))) + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))));
                    tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(64L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))) + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))));
                }
            }
        }
    }
    {
        {
            {
                auto tmp0 = static_cast<int64_t>(0);
                out_ptr2[static_cast<int64_t>(0L)] = tmp0;
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>((-1L) + ks2); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L))))))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))));
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp5 = static_cast<int64_t>(0);
                    auto tmp6 = at::vec::VectorizedN<int64_t,2>(tmp5);
                    auto tmp7 = at::vec::VecMask<int64_t,2>(tmp0 < tmp6);
                    auto tmp8 = decltype(tmp4)::blendv(tmp0, tmp4, tmp7.template cast<int64_t,2>());
                    auto tmp9 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp8.store(tmpbuf.data(), static_cast<int64_t>(16));
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp10 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(16); x0_inner++)
                        {
                            tmpbuf[x0_inner] = static_cast<int64_t>(tmp9[x0_inner]);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(16));
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp10) & (tmp10 < at::vec::VectorizedN<int64_t,2>(1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))).all_masked(), "index out of bounds: 0 <= tmp10 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))");
                    auto tmp12 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(16); x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr2[static_cast<int64_t>(tmp9[x0_inner])];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(16));
                    }
                    ()
                    ;
                    tmp12.store(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))) && x0 < static_cast<int64_t>((-1L) + ks2)))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))));
                    auto tmp1 = 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))));
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp5 = static_cast<int64_t>(0);
                    auto tmp6 = at::vec::VectorizedN<int64_t,2>(tmp5);
                    auto tmp7 = at::vec::VecMask<int64_t,2>(tmp0 < tmp6);
                    auto tmp8 = decltype(tmp4)::blendv(tmp0, tmp4, tmp7.template cast<int64_t,2>());
                    auto tmp9 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp8.store(tmpbuf.data(), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))));
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp10 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))); x0_inner++)
                        {
                            tmpbuf[x0_inner] = static_cast<int64_t>(tmp9[x0_inner]);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))));
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>::set(at::vec::VecMask<int64_t,2>::from(1), (at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp10) & (tmp10 < at::vec::VectorizedN<int64_t,2>(1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))))).all_masked(), "index out of bounds: 0 <= tmp10 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))");
                    auto tmp12 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))); x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr2[static_cast<int64_t>(tmp9[x0_inner])];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))));
                    }
                    ()
                    ;
                    tmp12.store(out_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>((-1L) + ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>((-1L) + ks2), static_cast<int64_t>(16L)))));
                }
            }
        }
    }
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(ks2); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L))))))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp13 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))));
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp5 = static_cast<int64_t>(0);
                    auto tmp6 = at::vec::VectorizedN<int64_t,2>(tmp5);
                    auto tmp7 = at::vec::VecMask<int64_t,2>(tmp0 < tmp6);
                    auto tmp8 = decltype(tmp4)::blendv(tmp0, tmp4, tmp7.template cast<int64_t,2>());
                    auto tmp9 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp8.store(tmpbuf.data(), static_cast<int64_t>(16));
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp10 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(16); x0_inner++)
                        {
                            tmpbuf[x0_inner] = static_cast<int64_t>(tmp9[x0_inner]);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(16));
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp10) & (tmp10 < at::vec::VectorizedN<int64_t,2>(1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))).all_masked(), "index out of bounds: 0 <= tmp10 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))");
                    auto tmp12 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(16); x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr2[static_cast<int64_t>(tmp9[x0_inner])];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(16));
                    }
                    ()
                    ;
                    auto tmp14 = tmp12 - tmp13;
                    tmp14.store(out_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))) && x0 < static_cast<int64_t>(ks2)))
                {
                    auto tmp0 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                    auto tmp13 = at::vec::VectorizedN<int64_t,2>::loadu(in_ptr3 + static_cast<int64_t>(x0), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                    auto tmp1 = 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))));
                    auto tmp2 = c10::convert<int64_t>(tmp1);
                    auto tmp3 = at::vec::VectorizedN<int64_t,2>(tmp2);
                    auto tmp4 = tmp0 + tmp3;
                    auto tmp5 = static_cast<int64_t>(0);
                    auto tmp6 = at::vec::VectorizedN<int64_t,2>(tmp5);
                    auto tmp7 = at::vec::VecMask<int64_t,2>(tmp0 < tmp6);
                    auto tmp8 = decltype(tmp4)::blendv(tmp0, tmp4, tmp7.template cast<int64_t,2>());
                    auto tmp9 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        tmp8.store(tmpbuf.data(), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                        return tmpbuf;
                    }
                    ()
                    ;
                    auto tmp10 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))); x0_inner++)
                        {
                            tmpbuf[x0_inner] = static_cast<int64_t>(tmp9[x0_inner]);
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                    }
                    ()
                    ;
                    TORCH_CHECK((at::vec::VecMask<int64_t,2>::set(at::vec::VecMask<int64_t,2>::from(1), (at::vec::VecMask<int64_t,2>((at::vec::VectorizedN<int64_t,2>(0) <= tmp10) & (tmp10 < at::vec::VectorizedN<int64_t,2>(1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))))))), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))))).all_masked(), "index out of bounds: 0 <= tmp10 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))");
                    auto tmp12 =
                    [&]
                    {
                        __at_align__ std::array<int64_t, 16> tmpbuf;
                        #pragma GCC unroll 16
                        for (long x0_inner = 0; x0_inner < static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))); x0_inner++)
                        {
                            tmpbuf[x0_inner] = in_ptr2[static_cast<int64_t>(tmp9[x0_inner])];
                        }
                        return at::vec::VectorizedN<int64_t,2>::loadu(tmpbuf.data(), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                    }
                    ()
                    ;
                    auto tmp14 = tmp12 - tmp13;
                    tmp14.store(out_ptr4 + static_cast<int64_t>(x0), static_cast<int64_t>(ks2 + (-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks2), static_cast<int64_t>(16L)))));
                }
            }
        }
    }
    inductor_cpu_integer_div_error_flag = nullptr;
    inductor_cpu_throw_if_integer_div_error(inductor_cpu_integer_div_error);
}
"""
)

inductor_entry = CppWrapperCodeCache.load_pybinding(
    argtypes=["std::vector<AtenTensorHandle>"],
    main_code=cpp_wrapper_src,
    device_type="cpu",
    needs_vec_isa=True,
    kernel_needs_vec_isa=None,
    num_outputs=1,
    kernel_code=None,
    extra_flags=(),
)

def _wrap_func(f):
    def g(args):
        input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg, device='cpu') for arg in args]
        input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)

        args.clear()
        del input_tensors

        output_handles = f(input_handles)
        output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
        return output_tensors

    return g

call = _wrap_func(inductor_entry)


def get_args():
    from torch._dynamo.testing import rand_strided
    arg0_1 = 6144
    arg1_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.uint8)
    arg2_1 = 4
    arg3_1 = 6144
    arg4_1 = rand_strided((6144, ), (1, ), device='cpu', dtype=torch.uint8)
    arg5_1 = 1536
    arg6_1 = rand_strided((1536, ), (1, ), device='cpu', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
