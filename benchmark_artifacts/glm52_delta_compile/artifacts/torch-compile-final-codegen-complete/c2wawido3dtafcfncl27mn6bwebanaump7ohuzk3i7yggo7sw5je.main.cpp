
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

// Python bindings to call inductor_entry_cpp():
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <sstream>
#include <cstdlib>
#include <cerrno>

// This is defined in guards.cpp so we don't need to import PyTorch headers that are slooow.
// We manually link it below to workaround issues with fbcode build.
static void* (*_torchinductor_pyobject_tensor_data_ptr)(PyObject* obj);

template <typename T> static inline T parse_arg(PyObject* args, size_t n) {
    static_assert(std::is_pointer_v<T>, "arg type must be pointer or long");
    return static_cast<T>(_torchinductor_pyobject_tensor_data_ptr(PyTuple_GET_ITEM(args, n)));
}
template <> inline int64_t parse_arg<int64_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsSsize_t(PyTuple_GET_ITEM(args, n));
    if(result == -1 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return result;
}
template <> inline uintptr_t parse_arg<uintptr_t>(PyObject* args, size_t n) {
    auto result = PyLong_AsVoidPtr(PyTuple_GET_ITEM(args, n));
    if(result == reinterpret_cast<void*>(-1) && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected int arg");
    return reinterpret_cast<uintptr_t>(result);
}
template <> inline float parse_arg<float>(PyObject* args, size_t n) {
    auto result = PyFloat_AsDouble(PyTuple_GET_ITEM(args, n));
    if(result == -1.0 && PyErr_Occurred()) [[unlikely]]
        throw std::runtime_error("expected float arg");
    return static_cast<float>(result);
}


#include <torch/csrc/inductor/aoti_torch/c/shim.h>

static inline std::vector<AtenTensorHandle> unpack_tensor_handle_list(PyObject* pyvec) {
    std::vector<AtenTensorHandle> result;
    size_t result_len = PyList_GET_SIZE(pyvec);
    result.reserve(result_len);
    for (size_t i = 0; i < result_len; i++) {
        // AtenTensorHandle is essentially a pointer
        void* elem = PyCapsule_GetPointer(PyList_GET_ITEM(pyvec, i), NULL);
        result.push_back(reinterpret_cast<AtenTensorHandle>(elem));
    }
    return result;
}

static inline PyObject* pack_tensor_handle_list(const std::array<AtenTensorHandle, 2>& arr) {
    PyObject* result = PyList_New(2);
    for (size_t i = 0; i < 2; i++) {
        PyObject *elem =
            arr[i] == nullptr
                ? Py_NewRef(Py_None)
                // Store AtenTensorHandle as PyCapsulate
                : PyCapsule_New(reinterpret_cast<void*>(arr[i]), NULL, NULL);
        PyList_SET_ITEM(result, i, elem);
    }
    return result;
}

template <> inline std::vector<AtenTensorHandle> parse_arg<std::vector<AtenTensorHandle>>(PyObject* args, size_t n) {
    return unpack_tensor_handle_list(PyTuple_GET_ITEM(args, n));
}

PyObject* inductor_entry_cpp(std::vector<AtenTensorHandle>&& input_handles) {
    // For outputs, we only allocate an array to hold returned tensor handles,
    // not the actual output tensor storage.
    std::array<AtenTensorHandle, 2> output_handles{};
    try {
        inductor_entry_impl(input_handles.data(), output_handles.data());
        if (PyErr_Occurred()) {
            return nullptr;
        }
        return pack_tensor_handle_list(output_handles);
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}


static PyObject* inductor_entry_cpp_py(PyObject* self, PyObject* args) {
    try {
        if(!PyTuple_CheckExact(args)) [[unlikely]]
            throw std::runtime_error("tuple args required");
        if(PyTuple_GET_SIZE(args) != 1) [[unlikely]]
            throw std::runtime_error("requires 1 args");
        return inductor_entry_cpp(parse_arg<std::vector<AtenTensorHandle>>(args, 0));
    } catch(std::exception const& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch(...) {
        PyErr_SetString(PyExc_RuntimeError, "unhandled error");
        return nullptr;
    }
}

static PyMethodDef py_methods[] = {
    {"inductor_entry_cpp", inductor_entry_cpp_py, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef py_module =
    {PyModuleDef_HEAD_INIT, "inductor_entry_cpp", NULL, -1, py_methods};

PyMODINIT_FUNC PyInit_inductor_entry_cpp(void) {
    const char* str_addr = std::getenv("_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
    if(!str_addr) {
        PyErr_SetString(PyExc_RuntimeError, "_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR must be set");
        return nullptr;
    }

    char* endptr = nullptr;
    errno = 0;
    uintptr_t addr = std::strtoull(str_addr, &endptr, 10);
    if(errno != 0 || endptr == str_addr || addr == 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to parse _TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR");
        return nullptr;
    }
    _torchinductor_pyobject_tensor_data_ptr =
        reinterpret_cast<decltype(_torchinductor_pyobject_tensor_data_ptr)>(addr);
    PyObject* module = PyModule_Create(&py_module);
    if (module == NULL) {
        return NULL;
    }
    #ifdef Py_GIL_DISABLED
        PyUnstable_Module_SetGIL(module, Py_MOD_GIL_NOT_USED);
    #endif
    return module;
}
