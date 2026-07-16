//#define PY_SSIZE_T_CLEAN
//#define Py_LIMITED_API
#include <Python.h>
#include <stdio.h>

typedef short T;

#if DEBUG
#define Py_DECREF_bak Py_DECREF
void __delete(PyObject* X){
    //int* refcnt = (int*)(X);
    //*refcnt = 0;
    Py_DECREF_bak(X);
    if (Py_REFCNT(X) != 0)
        printf("%d\n",Py_REFCNT(X));
    if (Py_REFCNT(X) < 0)
        abort();
}
#undef Py_DECREF
#define Py_DECREF(X) __delete(X);
#endif

#include <algorithm>
#include <array>
#include <string>
#include <iostream>

#include <map>
#include <goopax_extra/types.hpp>
#include "gpu_reshaper.hpp"
#include "waterfall.hpp"

/* Kernels  */
#include "mykernels.hpp"

/*  Module  */
static struct PyModuleDef gpu_reshaper_module = {
    PyModuleDef_HEAD_INIT,
    "gpu_reshaper",
    "",
    0,
    0, // methods
    NULL,
    NULL,
    NULL,
    NULL
};
static PyMethodDef silence_py_methods[] = {
    {"host_query", silence_host_query, METH_VARARGS | METH_STATIC, ""},
    {NULL, NULL, 0, NULL}
};
static PyTypeObject silence_type_descr = SimdFunction_Type("silence","Sets all the samples to zero",sizeof(SimdFunction<silence>),silence_py_methods,SimdFunction_new<silence>,SimdFunction_init<silence>,SimdFunction_call<silence>,SimdFunction_free<silence>);
PyMODINIT_FUNC PyInit_gpu_reshaper(void){
    PyObject* mod = PyModule_Create(&gpu_reshaper_module);
    if (PyModule_AddType(mod, &silence_type_descr) < 0)
        abort();
    return mod;
}

#if !MODULE_LIBRARY
int main(){
    const char* arg1 = "../output.wav";

    PyImport_AppendInittab("gpu_reshaper", PyInit_gpu_reshaper);
    Py_InitializeEx(0);
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* sys_m = PyImport_ImportModule("sys");
    PyObject* path = PyObject_GetAttrString(sys_m,"path");
    PyObject* append = PyObject_GetAttrString(path,"append");
    PyObject* append_args = PyTuple_New(1);
    PyObject* directory = PyUnicode_FromString("../python-modules");
    PyTuple_SetItem(append_args, 0, directory);
    PyObject_CallObject(append,append_args);

    PyObject* test_m = PyImport_ImportModule("test");

    PyObject* test_wav_simd = PyObject_GetAttrString(test_m,"test_wav_simd");
    PyObject* test_wav_simd_args = PyTuple_New(1);
    PyObject* filename = PyUnicode_FromString(arg1);
    PyTuple_SetItem(test_wav_simd_args, 0, filename);
    PyObject_CallObject(test_wav_simd,test_wav_simd_args);

    PyGILState_Release(gstate);
#if DEBUG
    if (Py_FinalizeEx() < 0)
        abort();
#else
    Py_Finalize();
#endif
    return 0;
}
#endif

#if DEBUG
#undef Py_DECREF
#define Py_DECREF Py_DECREF_bak;
#endif