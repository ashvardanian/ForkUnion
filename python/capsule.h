/**
 *  @brief  The cross-extension C API ForkUnion publishes to other Python packages.
 *  @file   capsule.h
 *  @author Ash Vardanian
 *
 *  Python loads extensions with `RTLD_LOCAL`, so one extension cannot `dlsym` another's symbols - which is why
 *  StringZilla hands `stringzillas` a `PyCapsule` holding a struct of function pointers rather than linking natively.
 *  The same trick applies here, and buys something linking cannot: whoever imports `forkunion` gets the @b same pool,
 *  because a Python module is a process-wide singleton no matter how many packages import it.
 *
 *  Linking `libforkunion.so.3` directly stays supported and is the faster path - see `forkunion.get_library_dir()`.
 *  Prefer this capsule when a consumer wants the @b shared pool rather than its own, or wants to avoid the
 *  `auditwheel` vendoring that silently mints a second copy of the library.
 *
 *  @code{.c}
 *  fu_capsule_v3_t const *api = (fu_capsule_v3_t const *)PyCapsule_Import(FU_CAPSULE_NAME_V3, 0);
 *  if (!api || api->abi_version != 3) { PyErr_Clear(); } // ! Fall back to a private pool
 *  else {
 *      Py_BEGIN_ALLOW_THREADS
 *      api->for_slices(api->pool, n, &my_kernel, &my_context);
 *      Py_END_ALLOW_THREADS
 *  }
 *  @endcode
 */
#ifndef FORKUNION_CAPSULE_H_
#define FORKUNION_CAPSULE_H_

#include <forkunion.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief The capsule's name, versioned so an ABI break fails the import instead of mis-calling. */
#define FU_CAPSULE_NAME_V3 "forkunion._C_API_v3"

/**
 *  @brief A struct of function pointers, published as one capsule.
 *
 *  One capsule holding a struct, rather than one capsule per function: a single `PyCapsule_Import` and a single
 *  version to check. `abi_version` is first so a future layout can still be recognized before anything else is read -
 *  StringZilla's `_sz_py_api` has no such field, and cannot tell a v1 consumer from a v2 producer.
 */
typedef struct fu_capsule_v3_t {
    /** @brief Always 3 for this layout. Check it before touching any field below. */
    unsigned abi_version;
    unsigned reserved;

    /** @brief The process-wide topology, owned by the `forkunion` module. Do not delete. */
    fu_topology_t topology;
    /** @brief The process-wide pool, owned by the `forkunion` module. Do not delete or terminate. */
    fu_pool_t pool;

    /*  The dispatch entry points. Each spawns the shared pool on first use, so importing `forkunion` costs no threads
     *  until someone actually dispatches - `import numpy` already spends 64 threads on a 128-core host, and an import
     *  that quietly does the same is not a good neighbor.
     *
     *  These serialize against each other: two Python threads that both drop the GIL would otherwise dispatch into one
     *  pool at once. The loser of the arbitration runs the loop itself, which is what "no nested parallelism" means
     *  here, and costs one uncontended atomic per dispatch.
     */
    void (*for_threads)(fu_pool_t, fu_for_threads_t, fu_lambda_context_t);
    void (*for_n)(fu_pool_t, size_t, fu_for_prongs_t, fu_lambda_context_t);
    void (*for_n_dynamic)(fu_pool_t, size_t, fu_for_prongs_t, fu_lambda_context_t);
    void (*for_slices)(fu_pool_t, size_t, fu_for_slices_t, fu_lambda_context_t);

    /** @brief Threads in the shared pool; zero before the first dispatch, and in a `fork()`-ed child. */
    size_t (*threads_count)(fu_pool_t);

} fu_capsule_v3_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // FORKUNION_CAPSULE_H_
