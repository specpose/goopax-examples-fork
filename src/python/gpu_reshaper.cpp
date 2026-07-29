#include <goopax_extra/types.hpp>
#include <stdio.h>

// #define PY_SSIZE_T_CLEAN
// #define Py_LIMITED_API
#include "gpu_buffer.cpp"

typedef short T;
#include "waterfall.hpp"
#include "simd_function.hpp"

#ifndef NDEBUG
#define Py_DECREF_bak Py_DECREF
void __delete(PyObject* X)
{
    // int* refcnt = (int*)(X);
    //*refcnt = 0;
    Py_DECREF_bak(X);
    if (Py_REFCNT(X) != 0)
        printf("%d\n", Py_REFCNT(X));
    if (Py_REFCNT(X) < 0)
        abort();
}
#undef Py_DECREF
#define Py_DECREF(X) __delete(X);
#endif

/* Kernels  */
#include "mykernels.hpp"

/*  Module  */
static struct PyModuleDef gpu_reshaper_module = { PyModuleDef_HEAD_INIT,
                                                  "gpu_reshaper",
                                                  "",
                                                  0,
                                                  0, // module methods
                                                  NULL,
                                                  NULL,
                                                  NULL,
                                                  NULL };
static PyMethodDef silence_py_methods[] = {
    { "host_query", (PyCFunction)BaseFunction_host_query<silence>, METH_VARARGS | METH_KEYWORDS | METH_STATIC, "" },
    { NULL, NULL, 0, NULL }
};
static PyTypeObject silence_type_descr = BaseFunction_Type("silence",
                                                           "Sets all the samples to zero",
                                                           sizeof(BaseFunction<silence>),
                                                           silence_py_methods,
                                                           BaseFunction_new<silence>,
                                                           BaseFunction_init<silence>,
                                                           BaseFunction_call<silence>,
                                                           BaseFunction_free<silence>);
PyMODINIT_FUNC PyInit_gpu_reshaper(void)
{
    PyObject* mod = PyModule_Create(&gpu_reshaper_module);
    if (PyModule_AddType(mod, &silence_type_descr) < 0)
        abort();
    return mod;
}

#if !MODULE_LIBRARY
int main(int argc, char* argv[])
{
    bool args[] = { true, false };
    std::string p;
    std::string arg1;
    if (argc < 2)
    {
        args[1] = true;
        p = "test_generator_simd";
        arg1 = std::string(argv[0]) + ".out";
    }
    else if (argc == 2)
    {
        args[1] = true;
        p = "test_wav_simd";
        arg1 = std::string(argv[1]);
    }
    else
    {
        std::cerr << "usage: " << argv[0] << " fileName" << std::endl;
        exit(1);
    }

    PyImport_AppendInittab("gpu_buffer", PyInit_gpu_buffer);
    PyImport_AppendInittab("gpu_reshaper", PyInit_gpu_reshaper);
    Py_InitializeEx(0);
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* sys_m = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys_m, "path");
    PyObject* append = PyObject_GetAttrString(path, "append");
    PyObject* append_args = PyTuple_New(1);
    PyObject* directory = PyUnicode_FromString("../../src/python/python-modules");
    PyTuple_SetItem(append_args, 0, directory);
    PyObject_CallObject(append, append_args);

    PyObject* test_m = PyImport_ImportModule("test");

    PyObject* prog = PyObject_GetAttrString(test_m, p.c_str());
    PyObject* prog_args = PyTuple_New(1);
    PyObject* filename = PyUnicode_FromString(arg1.c_str());
    PyTuple_SetItem(prog_args, 0, filename);
    PyObject_CallObject(prog, prog_args);

    PyGILState_Release(gstate);
    return Py_FinalizeEx();
}
#endif

#ifndef NDEBUG
#undef Py_DECREF
#define Py_DECREF Py_DECREF_bak;
#endif