

import torch
from torch._inductor.codecache import CppWrapperCodeCache

cpp_wrapper_src = (
r"""
#include <torch/csrc/inductor/cpp_wrapper/cpu.h>
extern "C"  void  cpp_fused_bitwise_xor_clone_view_0(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       const int64_t ks0);
CACHE_TORCH_DTYPE(int32);
CACHE_TORCH_DTYPE(uint8);
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
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 3);
    int64_t arg0_1;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_int64(inputs[0], &arg0_1));
    auto arg1_1 = std::move(inputs[1]);
    auto arg2_1 = std::move(inputs[2]);
    int64_t s65 = arg0_1;
    assert_size_stride(arg1_1, {s65, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [new_words], Original ATen: [aten.view]
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg1_1, cached_torch_dtype_int32, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    assert_size_stride(buf1, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
    assert_size_stride(arg2_1, {s65, }, {1L, }, "input");
    // Topologically Sorted Source Nodes: [old_words], Original ATen: [aten.view]
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(arg2_1, cached_torch_dtype_int32, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    assert_size_stride(buf3, {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), }, {1L, }, "torch.ops.aten.view.dtype");
    const int64_t int_array_0[] = {c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L)), };
    static constexpr int64_t int_array_1[] = {1L, };
    AtenTensorHandle buf4_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_int32, cached_torch_device_type_cpu, 0, &buf4_handle));
    RAIIAtenTensorHandle buf4(buf4_handle);
    AtenTensorHandle buf7_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_0, int_array_1, cached_torch_dtype_int32, cached_torch_device_type_cpu, 0, &buf7_handle));
    RAIIAtenTensorHandle buf7(buf7_handle);
    cpp_fused_bitwise_xor_clone_view_0((const int32_t*)(buf1.data_ptr()), (const int32_t*)(buf3.data_ptr()), (int32_t*)(buf4.data_ptr()), (int32_t*)(buf7.data_ptr()), s65);
    arg1_1.reset();
    arg2_1.reset();

    buf1.reset();

    buf3.reset();
    // Topologically Sorted Source Nodes: [xor, view_2], Original ATen: [aten.bitwise_xor, aten.view]
    AtenTensorHandle buf6_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(buf4, cached_torch_dtype_uint8, &buf6_handle));
    RAIIAtenTensorHandle buf6(buf6_handle);
    assert_size_stride(buf6, {4L*(c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L))), }, {1L, }, "torch.ops.aten.view.dtype");
    // Topologically Sorted Source Nodes: [clone, view_3], Original ATen: [aten.clone, aten.view]
    AtenTensorHandle buf9_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_view_dtype(buf7, cached_torch_dtype_uint8, &buf9_handle));
    RAIIAtenTensorHandle buf9(buf9_handle);
    assert_size_stride(buf9, {4L*(c10::div_floor_integer(static_cast<int64_t>(s65), static_cast<int64_t>(4L))), }, {1L, }, "torch.ops.aten.view.dtype");
    output_handles[0] = buf6.release();
    output_handles[1] = buf9.release();
} // inductor_entry_impl

#include <torch/csrc/inductor/cpp_prefix.h>
extern "C"  void  cpp_fused_bitwise_xor_clone_view_0(const int32_t* in_ptr0,
                       const int32_t* in_ptr1,
                       int32_t* out_ptr0,
                       int32_t* out_ptr1,
                       const int64_t ks0)
{
    std::atomic<int> inductor_cpu_integer_div_error{0};
    inductor_cpu_integer_div_error_flag = &inductor_cpu_integer_div_error;
    {
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L))); x0+=static_cast<int64_t>(16L))
        {
            {
                if(C10_LIKELY(x0 >= static_cast<int64_t>(0) && x0 < static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))))))
                {
                    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp1 = at::vec::Vectorized<int32_t>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    auto tmp2 = tmp0 ^ tmp1;
                    tmp2.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                    tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>(16));
                }
                if(C10_UNLIKELY(x0 >= static_cast<int64_t>(16L*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L)))) && x0 < static_cast<int64_t>(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))))
                {
                    auto tmp0 = at::vec::Vectorized<int32_t>::loadu(in_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                    auto tmp1 = at::vec::Vectorized<int32_t>::loadu(in_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                    auto tmp2 = tmp0 ^ tmp1;
                    tmp2.store(out_ptr0 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
                    tmp0.store(out_ptr1 + static_cast<int64_t>(x0), static_cast<int64_t>((-16L)*(c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(64L))) + (c10::div_floor_integer(static_cast<int64_t>(ks0), static_cast<int64_t>(4L)))));
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
    num_outputs=2,
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
    arg2_1 = rand_strided((1903165440, ), (1, ), device='cpu', dtype=torch.uint8)
    return [arg0_1, arg1_1, arg2_1]


def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cpu')


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    args = get_args()
    compiled_module_main('None', lambda times, repeat: benchmark_compiled_module(args, times=times, repeat=repeat))
