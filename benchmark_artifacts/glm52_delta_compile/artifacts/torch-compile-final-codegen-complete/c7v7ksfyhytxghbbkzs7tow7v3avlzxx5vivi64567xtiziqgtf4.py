

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
extern "C"  void  cpp_fused_cat_new_zeros_sub_1(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       const int64_t ks0,
                       const int64_t ks1);
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
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 5);
    int64_t arg0_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[0], &arg0_1));
    auto arg1_1 = std::move(inputs[1]);
    int64_t arg2_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[2], &arg2_1));
    auto arg3_1 = std::move(inputs[3]);
    auto arg4_1 = std::move(inputs[4]);
    int64_t s65 = arg0_1;
    int64_t s37 = arg2_1;
    assert_size_stride(arg1_1, {s65, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [view], Original ATen: [aten.view]
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg1_1, cached_torch_dtype_int32, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    assert_size_stride(buf1, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
    assert_size_stride(arg3_1, {s65, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [view_2], Original ATen: [aten.view]
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg3_1, cached_torch_dtype_int32, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    assert_size_stride(buf3, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
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
    arg3_1.reset();

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
    assert_size_stride(arg4_1, {1L, }, {1L, }, "input");
    AtenTensorHandle buf11_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_1, int_array_1, cached_torch_dtype_int64, cached_torch_device_type_cpu, 0, &buf11_handle));
    RAIIAtenTensorHandle buf11(buf11_handle);
    cpp_fused_cat_new_zeros_sub_1((const int64_t*)(buf7.data_ptr()), (const int64_t*)(arg4_1.data_ptr()), (const int64_t*)(buf10.data_ptr()), (int64_t*)(buf8.data_ptr()), (int64_t*)(buf9.data_ptr()), (int64_t*)(buf11.data_ptr()), s37, s65);
    arg4_1.reset();
    output_handles[0] = buf11.release();
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
extern "C"  void  cpp_fused_cat_new_zeros_sub_1(const int64_t* in_ptr0,
                       const int64_t* in_ptr1,
                       const int64_t* in_ptr2,
                       int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       const int64_t ks0,
                       const int64_t ks1)
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
                auto tmp0 = in_ptr1[static_cast<int64_t>(0L)];
                auto tmp1 = 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))));
                auto tmp2 = c10::convert<int64_t>(tmp1);
                auto tmp3 = int64_t(tmp0 + tmp2);
                auto tmp4 = tmp0 < 0;
                auto tmp5 = tmp4 ? tmp3 : tmp0;
                auto tmp6 = tmp5;
                auto tmp7 = c10::convert<int64_t>(tmp6);
                TORCH_CHECK((0 <= tmp7) & (tmp7 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))), "index out of bounds: 0 <= tmp7 < 1L + (c10::div_floor_integer(static_cast<int64_t>(ks1), static_cast<int64_t>(4L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))))))");
                auto tmp9 = in_ptr2[static_cast<int64_t>(tmp5)];
                out_ptr2[static_cast<int64_t>(0L)] = tmp9;
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
    arg0_1 = 1903165440
    arg1_1 = rand_strided((1903165440, ), (1, ), device='cpu', dtype=torch.uint8)
    arg2_1 = 4096
    arg3_1 = rand_strided((1903165440, ), (1, ), device='cpu', dtype=torch.uint8)
    arg4_1 = rand_strided((1, ), (1, ), device='cpu', dtype=torch.int64)
    return [arg0_1, arg1_1, arg2_1, arg3_1, arg4_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
