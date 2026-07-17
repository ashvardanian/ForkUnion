/**
 *  @brief  The `forkunion` Python extension: owns one pool per process and publishes it.
 *  @file   _core.c
 *  @author Ash Vardanian
 *
 *  This module exists to be a singleton. A Python module is imported once per interpreter no matter how many packages
 *  ask for it, so a pool that lives here is shared by every consumer that imports `forkunion` - which is the whole
 *  point. Today a process running `numpy`, `sklearn`, `xgboost`, and `lightgbm` carries five independent pools
 *  declaring 512 threads on 128 cores, and nothing in the ecosystem can unify them.
 *
 *  @sa `capsule.h` for the published C API, `__init__.py` for `get_include()` / `get_library_dir()`.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "capsule.h"

/*  The pool is process-wide, so its guards are too. `spawn_state_` is 0 before the first dispatch, 1 once spawned, and
 *  -1 if the spawn failed and should not be retried on every call.
 */
static fu_capsule_v3_t api_;
static int spawn_state_ = 0;
static PyThread_type_lock spawn_lock_ = NULL;

/*  Held across a dispatch. ForkUnion bans nested parallelism and a pool runs one generation at a time, but the GIL
 *  cannot enforce that here: a dispatch releases it, so two Python threads can arrive at once. The loser runs the loop
 *  itself rather than blocking - refusing to nest is also exactly what libgomp does by default.
 */
static PyThread_type_lock dispatch_lock_ = NULL;

/** @brief Spawns the shared pool on first use. Returns 0 on failure. */
static int ensure_spawned_(void) {
    if (spawn_state_ == 1) return 1;
    if (spawn_state_ == -1) return 0;

    PyThread_acquire_lock(spawn_lock_, WAIT_LOCK);
    if (spawn_state_ == 0) {
        // ! Zero does not mean "every core" - it fails the spawn - so ask the topology for the count.
        size_t const threads = fu_logical_cores_count(api_.topology);
        int const ok = threads != 0 && fu_pool_spawn(api_.topology, api_.pool, threads, fu_caller_inclusive_k) != 0;
        spawn_state_ = ok ? 1 : -1;
    }
    PyThread_release_lock(spawn_lock_);
    return spawn_state_ == 1;
}

/** @brief True if this thread claimed the pool; false means somebody else is dispatching and we must go serial. */
static int claim_pool_(void) { return PyThread_acquire_lock(dispatch_lock_, NOWAIT_LOCK); }
static void release_pool_(void) { PyThread_release_lock(dispatch_lock_); }

static void for_threads_(fu_pool_t pool, fu_for_threads_t callback, fu_lambda_context_t context) {
    if (!ensure_spawned_() || !claim_pool_()) return callback(context, 0, 0);
    fu_pool_for_threads(pool, callback, context);
    release_pool_();
}

static void for_n_(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    if (!ensure_spawned_() || !claim_pool_()) {
        for (size_t i = 0; i != n; ++i) callback(context, i, 0, 0);
        return;
    }
    fu_pool_for_n(pool, n, callback, context);
    release_pool_();
}

static void for_n_dynamic_(fu_pool_t pool, size_t n, fu_for_prongs_t callback, fu_lambda_context_t context) {
    if (!ensure_spawned_() || !claim_pool_()) {
        for (size_t i = 0; i != n; ++i) callback(context, i, 0, 0);
        return;
    }
    fu_pool_for_n_dynamic(pool, n, callback, context);
    release_pool_();
}

static void for_slices_(fu_pool_t pool, size_t n, fu_for_slices_t callback, fu_lambda_context_t context) {
    if (!ensure_spawned_() || !claim_pool_()) return callback(context, 0, n, 0, 0);
    fu_pool_for_slices(pool, n, callback, context);
    release_pool_();
}

static size_t threads_count_(fu_pool_t pool) { return spawn_state_ == 1 ? fu_pool_threads_count(pool) : 0; }

#pragma region Python API

static PyObject *py_threads(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    return PyLong_FromSize_t(threads_count_(api_.pool));
}

static PyObject *py_spawn(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    if (!ensure_spawned_()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to spawn the ForkUnion pool");
        return NULL;
    }
    return PyLong_FromSize_t(fu_pool_threads_count(api_.pool));
}

static PyObject *py_compute_domains(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    return PyLong_FromSize_t(fu_compute_domains_count(api_.topology));
}

static PyObject *py_memory_domains(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    return PyLong_FromSize_t(fu_memory_domains_count(api_.topology));
}

static PyObject *py_logical_cores(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    return PyLong_FromSize_t(fu_logical_cores_count(api_.topology));
}

static PyObject *py_capabilities(PyObject *self, PyObject *args) {
    (void)self, (void)args;
    char buffer[256];
    fu_name_capabilities(fu_runtime_capabilities(), buffer, sizeof(buffer));
    return PyUnicode_FromString(buffer);
}

static PyMethodDef methods_[] = {
    {"threads", py_threads, METH_NOARGS, "Threads in the shared pool; 0 before the first dispatch."},
    {"spawn", py_spawn, METH_NOARGS, "Spawn the shared pool now instead of on first dispatch."},
    {"compute_domains", py_compute_domains, METH_NOARGS, "Compute domains the machine reports."},
    {"memory_domains", py_memory_domains, METH_NOARGS, "Memory domains - NUMA nodes - the machine reports."},
    {"logical_cores", py_logical_cores, METH_NOARGS, "Logical cores visible to this process."},
    {"capabilities", py_capabilities, METH_NOARGS, "Capabilities detected on the running host."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_ = {PyModuleDef_HEAD_INIT, "forkunion._core",
                                     "The process-wide ForkUnion pool and its cross-extension C API.", -1, methods_};

PyMODINIT_FUNC PyInit__core(void) {
    PyObject *module = PyModule_Create(&module_);
    if (!module) return NULL;

    spawn_lock_ = PyThread_allocate_lock();
    dispatch_lock_ = PyThread_allocate_lock();
    if (!spawn_lock_ || !dispatch_lock_) {
        Py_DECREF(module);
        return PyErr_NoMemory();
    }

    api_.abi_version = 3;
    api_.reserved = 0;
    api_.topology = fu_topology_new();
    if (!api_.topology) {
        Py_DECREF(module);
        PyErr_SetString(PyExc_RuntimeError, "Failed to detect the hardware topology");
        return NULL;
    }
    api_.pool = fu_pool_new("python", fu_capabilities_all_k);
    if (!api_.pool) {
        Py_DECREF(module);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create the ForkUnion pool");
        return NULL;
    }

    // ! Deliberately no `fu_pool_spawn` here: importing must not cost threads.
    api_.for_threads = &for_threads_;
    api_.for_n = &for_n_;
    api_.for_n_dynamic = &for_n_dynamic_;
    api_.for_slices = &for_slices_;
    api_.threads_count = &threads_count_;

    // ! The attribute name must match the capsule's own name: `PyCapsule_Import("forkunion._C_API_v3")` imports the
    // ! `forkunion` package and reads `_C_API_v3` off it, so `__init__.py` re-exports this from `_core`.
    PyObject *capsule = PyCapsule_New(&api_, FU_CAPSULE_NAME_V3, NULL);
    if (!capsule || PyModule_AddObject(module, "_C_API_v3", capsule) < 0) {
        Py_XDECREF(capsule);
        Py_DECREF(module);
        return NULL;
    }
    if (PyModule_AddIntConstant(module, "abi_version", 3) < 0) {
        Py_DECREF(module);
        return NULL;
    }
    return module;
}

#pragma endregion Python API
