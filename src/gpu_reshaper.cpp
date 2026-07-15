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

    PyObject* import_wav_m = PyImport_ImportModule("import_wav");

    PyObject* get_format = PyObject_GetAttrString(import_wav_m,"wav_format");
    PyObject* get_format_args = PyTuple_New(1);
    PyObject* filename = PyUnicode_FromString(arg1);
    PyTuple_SetItem(get_format_args, 0, filename);
    PyObject* wav_format = PyObject_CallObject(get_format,get_format_args);
    PyObject_Print(wav_format,stdout,Py_PRINT_RAW);

    PyObject* gpu_reshaper_m = PyImport_ImportModule("gpu_reshaper");
    PyObject* silence = PyObject_GetAttrString(gpu_reshaper_m,"silence");
    PyObject* descriptor = PyObject_CallMethod(silence,"host_query","O",wav_format);
    printf("Dumping memoryview descriptor:\n");
    PyObject_Print(descriptor,stdout,Py_PRINT_RAW);
    printf("\n");

    if (PyList_Size(descriptor)>0){
        PyObject* device_selector_m = PyImport_ImportModule("device_selector");
        PyObject* select = PyObject_GetAttrString(device_selector_m,"retrieve_shape");
        PyObject* select_args = PyTuple_New(1);
        PyTuple_SetItem(select_args, 0, descriptor);
        PyObject* selection = PyObject_CallObject(select,select_args);
        PyObject_Print(selection,stdout,Py_PRINT_RAW);

        PyObject* import_data = PyObject_GetAttrString(import_wav_m,"import_wav_aligned");
        PyObject* import_data_args = PyTuple_New(3);
        PyTuple_SetItem(import_data_args, 0, filename);
        PyTuple_SetItem(import_data_args, 1, wav_format);
        PyTuple_SetItem(import_data_args, 2, selection);
        PyObject* custom_buffer = PyObject_CallObject(import_data,import_data_args);
        printf("Imported data from file \"%s\".\n", arg1);

        //PyObject* shape = PyTuple_GetItem(selection, 2);
        //PyObject* reshape = PyObject_GetAttrString(numpy_array,"reshape");
        //printf("Python ndarray reshaping to:\n");
        //PyObject_Print(shape,stdout,Py_PRINT_RAW);
        //printf("\n");
        //PyObject* shape_args = PyTuple_New(1);
        //PyTuple_SetItem(shape_args, 0, shape);
        //numpy_array = PyObject_CallObject(reshape, shape_args);

        PyObject* Initialisation = PyTuple_New(2);
        PyTuple_SetItem(Initialisation, 0, PyTuple_GetItem(selection, 0));
        PyTuple_SetItem(Initialisation, 1, PyTuple_GetItem(selection, 1));
        PyObject* Custom_Callable = PyObject_CallObject(silence, Initialisation);

        //Py_buffer* view;
        //if (PyObject_GetBuffer(numpy_array, view, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT)==0){
        PyObject* process_args = PyTuple_New(1);
        PyObject* mv = PyMemoryView_GetContiguous(custom_buffer, PyBUF_WRITE, 'C');
        //PyObject* mv = PyMemoryView_FromObject(custom_buffer);
        //PyObject* mv = PyMemoryView_FromBuffer(view);
        PyTuple_SetItem(process_args, 0, mv);
        PyObject* result = PyObject_CallObject(Custom_Callable, process_args);
        PyObject_Del((void*)Custom_Callable);
        //PyBuffer_Release(view);
        //}

        printf("Exporting data to file \"%s\".\n", arg1);
        PyObject* export_data = PyObject_GetAttrString(import_wav_m,"export_data");
        PyObject* export_data_args = PyTuple_New(2);
        PyTuple_SetItem(export_data_args, 0, filename);
        Py_INCREF(custom_buffer);
        PyTuple_SetItem(export_data_args, 1, custom_buffer);
        PyObject_CallObject(export_data,export_data_args);
    }

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