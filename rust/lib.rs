//! Low-latency OpenMP-style NUMA-aware cross-platform fine-grained parallelism library.
//!
//! Fork Union provides a minimalistic cross-platform thread-pool implementation and Parallel Algorithms,
//! avoiding dynamic memory allocations, exceptions, system calls, and heavy Compare-And-Swap instructions.
//! The library leverages the "weak memory model" to allow Arm and IBM Power CPUs to aggressively optimize
//! execution at runtime. It also aggressively tests against overflows on smaller index types, and is safe
//! to use even with the maximal `usize` values.
//!
//! This Rust wrapper provides a safe interface around the precompiled C library, maintaining zero-allocation
//! principles while leveraging NUMA-aware optimizations and CPU-specific busy-waiting instructions.

#![no_std]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::ffi::CStr;

use core::cell::UnsafeCell;
use core::ffi::{c_char, c_int, c_void};
use core::marker::PhantomData;
use core::ptr::NonNull;
use core::slice;
use core::sync::atomic::{AtomicBool, Ordering};

/// Default alignment for preventing false sharing between threads.
///
/// Set to 128 bytes to account for adjacent cache-line prefetching on modern CPUs.
/// This matches the C++ `default_alignment_k` constant defined in `fork_union.hpp`.
///
/// On x86, most CPUs fetch 2 cache lines (128 bytes) at once with spatial prefetching enabled.
/// This conservative padding prevents false sharing even with aggressive prefetch settings.
pub const DEFAULT_ALIGNMENT: usize = 128;

/// Cache-line aligned wrapper to prevent false sharing between threads.
///
/// When multiple threads access separate data that resides on the same cache line,
/// modifications by one thread invalidate the cache line for all others, causing
/// performance degradation known as "false sharing".
///
/// This wrapper ensures each wrapped value occupies its own cache line (128 bytes),
/// eliminating false sharing at the cost of increased memory usage.
///
/// # Examples
///
/// ```rust
/// use fork_union::{CacheAligned, ThreadPool};
///
/// let mut pool = ThreadPool::try_spawn(4).unwrap();
/// let data: Vec<usize> = (0..1000).collect();
///
/// // Each thread gets its own cache-aligned accumulator
/// let mut scratch: Vec<CacheAligned<usize>> =
///     (0..pool.threads()).map(|_| CacheAligned(0)).collect();
///
/// // No false sharing during parallel reduction
/// for value in &data {
///     let tid = *value % pool.threads();
///     scratch[tid].0 += value;
/// }
///
/// let total: usize = scratch.iter().map(|a| a.0).sum();
/// ```
#[repr(align(128))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheAligned<T>(pub T);

// Compile-time assertion that alignment matches DEFAULT_ALIGNMENT
const _: () = assert!(
    core::mem::align_of::<CacheAligned<u8>>() == DEFAULT_ALIGNMENT,
    "CacheAligned alignment must match DEFAULT_ALIGNMENT"
);

/// A generic spin mutex that uses CPU-specific pause instructions for efficient busy-waiting.
///
/// This is a low-level synchronization primitive that spins on a busy loop rather than
/// blocking the thread. It's most appropriate for very short critical sections where
/// the cost of context switching would be higher than busy-waiting.
///
/// The generic parameter `P` allows customization of the pause behavior:
/// - `true` enables CPU-specific pause instructions (recommended for most use cases)
/// - `false` disables pause instructions (may be useful in some specialized scenarios)
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// // Create a spin mutex with pause instructions enabled
/// let mutex = BasicSpinMutex::<i32, true>::new(42);
///
/// // Lock, access data, and unlock
/// {
///     let mut guard = mutex.lock();
///     *guard = 100;
/// } // Lock is automatically released when guard goes out of scope
///
/// // Verify the value was changed
/// assert_eq!(*mutex.lock(), 100);
/// ```
///
/// Fast for short critical sections but spins continuously. Use when latency matters
/// more than CPU usage. Avoid for long critical sections or high contention scenarios.
pub struct BasicSpinMutex<T, const PAUSE: bool> {
    locked: AtomicBool,
    data: UnsafeCell<T>,
}

impl<T, const PAUSE: bool> BasicSpinMutex<T, PAUSE> {
    /// Creates a new spin mutex in the unlocked state.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to be protected by the mutex
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// ```
    pub const fn new(data: T) -> Self {
        Self {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }

    /// Acquires the lock, returning a guard that provides access to the protected data.
    ///
    /// This method will spin until the lock is acquired. If the lock is already held,
    /// it will busy-wait using CPU-specific pause instructions (if `PAUSE = true`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// let mut guard = mutex.lock();
    /// *guard = 42;
    /// ```
    pub fn lock(&self) -> BasicSpinMutexGuard<'_, T, PAUSE> {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Busy-wait with pause instructions if enabled
            if PAUSE {
                core::hint::spin_loop();
            }
        }
        BasicSpinMutexGuard { mutex: self }
    }

    /// Attempts to acquire the lock without blocking.
    ///
    /// Returns `Some(guard)` if the lock was successfully acquired, or `None` if
    /// the lock is currently held by another thread.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    ///
    /// if let Some(mut guard) = mutex.try_lock() {
    ///     *guard = 42;
    ///     println!("Lock acquired and value set");
    /// } else {
    ///     println!("Lock is currently held by another thread");
    /// };
    /// ```
    pub fn try_lock(&self) -> Option<BasicSpinMutexGuard<'_, T, PAUSE>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(BasicSpinMutexGuard { mutex: self })
        } else {
            None
        }
    }

    /// Checks if the mutex is currently locked.
    ///
    /// This method provides a non-blocking way to check the lock state, but should
    /// be used carefully as the state can change immediately after this call returns.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(0);
    /// assert!(!mutex.is_locked());
    ///
    /// {
    ///     let _guard = mutex.lock();
    ///     assert!(mutex.is_locked());
    /// }
    ///
    /// assert!(!mutex.is_locked());
    /// ```
    pub fn is_locked(&self) -> bool {
        self.locked.load(Ordering::Acquire)
    }

    /// Consumes the mutex and returns the protected data.
    ///
    /// This method bypasses the locking mechanism entirely since we have exclusive
    /// ownership of the mutex.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mutex = BasicSpinMutex::<i32, true>::new(42);
    /// let data = mutex.into_inner();
    /// assert_eq!(data, 42);
    /// ```
    pub fn into_inner(self) -> T {
        self.data.into_inner()
    }

    /// Gets a mutable reference to the protected data.
    ///
    /// Since this requires a mutable reference to the mutex, no locking is needed
    /// as we have exclusive access.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut mutex = BasicSpinMutex::<i32, true>::new(0);
    /// *mutex.get_mut() = 42;
    /// assert_eq!(*mutex.lock(), 42);
    /// ```
    pub fn get_mut(&mut self) -> &mut T {
        self.data.get_mut()
    }
}

// Safety: BasicSpinMutex can be sent between threads if T can be sent
unsafe impl<T: Send, const PAUSE: bool> Send for BasicSpinMutex<T, PAUSE> {}
// Safety: BasicSpinMutex can be shared between threads if T can be sent
unsafe impl<T: Send, const PAUSE: bool> Sync for BasicSpinMutex<T, PAUSE> {}

/// A guard providing access to the data protected by a `BasicSpinMutex`.
///
/// The lock is automatically released when this guard is dropped.
pub struct BasicSpinMutexGuard<'a, T, const PAUSE: bool> {
    mutex: &'a BasicSpinMutex<T, PAUSE>,
}

impl<'a, T, const PAUSE: bool> BasicSpinMutexGuard<'a, T, PAUSE> {
    /// Returns a reference to the protected data.
    ///
    /// This method is rarely needed since the guard implements `Deref`.
    pub fn get(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }

    /// Returns a mutable reference to the protected data.
    ///
    /// This method is rarely needed since the guard implements `DerefMut`.
    pub fn get_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> core::ops::Deref for BasicSpinMutexGuard<'a, T, PAUSE> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> core::ops::DerefMut for BasicSpinMutexGuard<'a, T, PAUSE> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.mutex.data.get() }
    }
}

impl<'a, T, const PAUSE: bool> Drop for BasicSpinMutexGuard<'a, T, PAUSE> {
    fn drop(&mut self) {
        self.mutex.locked.store(false, Ordering::Release);
    }
}

/// A type alias for the most commonly used spin mutex configuration.
///
/// This is equivalent to `BasicSpinMutex<T, true>`, which enables CPU-specific
/// pause instructions for efficient busy-waiting.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let mutex = SpinMutex::new(42);
/// let mut guard = mutex.lock();
/// *guard = 100;
/// ```
pub type SpinMutex<T> = BasicSpinMutex<T, true>;

/// A "prong" - the tip of a "fork" - pinning a "task" to a "thread" and "memory" location.
///
/// A `Prong` represents a single unit of work that connects:
/// - A **task** (what work to do) - identified by `task_index`  
/// - A **thread** (which CPU thread is executing it) - identified by `thread_index`
/// - A **colocation** (which NUMA node/QoS level it's running on) - identified by `colocation_index`
///
/// This metadata is essential for NUMA-aware algorithms, debugging parallel execution,
/// and understanding load distribution across the thread pool.
#[derive(Copy, Clone, Debug)]
pub struct Prong {
    /// The logical index of the task being processed (0-based)
    pub task_index: usize,
    /// The physical thread executing this task (0-based)  
    pub thread_index: usize,
    /// The colocation group this thread belongs to (NUMA node + QoS level)
    pub colocation_index: usize,
}

/// Error types that can occur during thread pool operations.
#[derive(Debug)]
pub enum Error {
    /// Thread pool creation failed
    CreationFailed,
    /// Thread spawning failed
    SpawnFailed,
    /// Invalid parameter provided
    InvalidParameter,
    /// Platform not supported
    UnsupportedPlatform,
}

#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CreationFailed => write!(f, "failed to create thread pool"),
            Self::SpawnFailed => write!(f, "failed to spawn worker threads"),
            Self::InvalidParameter => write!(f, "invalid parameter provided"),
            Self::UnsupportedPlatform => write!(f, "platform not supported"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

// C FFI declarations
extern "C" {

    // Library metadata
    fn fu_version_major() -> c_int;
    fn fu_version_minor() -> c_int;
    fn fu_version_patch() -> c_int;
    fn fu_enabled_numa() -> c_int;
    fn fu_capabilities_string() -> *const c_char;

    // Systems metadata
    fn fu_count_logical_cores() -> usize;
    fn fu_count_colocations() -> usize;
    fn fu_count_numa_nodes() -> usize;
    fn fu_count_quality_levels() -> usize;
    fn fu_volume_any_pages() -> usize;

    // Core thread pool operations
    fn fu_pool_new(name: *const c_char) -> *mut c_void;
    fn fu_pool_delete(pool: *mut c_void);
    fn fu_pool_spawn(pool: *mut c_void, threads: usize, exclusivity: c_int) -> c_int;
    fn fu_pool_terminate(pool: *mut c_void);
    fn fu_pool_count_threads(pool: *mut c_void) -> usize;
    fn fu_pool_count_colocations(pool: *mut c_void) -> usize;
    fn fu_pool_count_threads_in(pool: *mut c_void, colocation_index: usize) -> usize;
    fn fu_pool_locate_thread_in(
        pool: *mut c_void,
        global_thread_index: usize,
        colocation_index: usize,
    ) -> usize;

    #[allow(dead_code)]
    fn fu_pool_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_n(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_n_dynamic(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_slices(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize, usize),
        context: *mut c_void,
    );

    // Advanced control flow
    fn fu_pool_unsafe_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_unsafe_join(pool: *mut c_void);
    fn fu_pool_sleep(pool: *mut c_void, micros: usize);

    // Memory management and NUMA
    fn fu_allocate_at_least(
        numa_node_index: usize,
        minimum_bytes: usize,
        allocated_bytes: *mut usize,
        bytes_per_page: *mut usize,
    ) -> *mut c_void;
    fn fu_allocate(numa_node_index: usize, bytes: usize) -> *mut c_void;
    fn fu_free(numa_node_index: usize, pointer: *mut c_void, bytes: usize);
    fn fu_volume_huge_pages_in(numa_node_index: usize) -> usize;
    fn fu_volume_any_pages_in(numa_node_index: usize) -> usize;

}

/// Returns a string describing available platform capabilities.
#[cfg(feature = "std")]
pub fn capabilities_string() -> Option<&'static str> {
    unsafe {
        let ptr = fu_capabilities_string();
        if ptr.is_null() {
            None
        } else {
            CStr::from_ptr(ptr).to_str().ok()
        }
    }
}

/// Returns a raw pointer to the capabilities string for no_std environments.
pub fn capabilities_string_ptr() -> *const c_char {
    unsafe { fu_capabilities_string() }
}

/// Returns the total volume of any pages (huge or regular) available across all NUMA nodes.
pub fn volume_any_pages() -> usize {
    unsafe { fu_volume_any_pages() }
}

/// Returns the number of logical CPU cores available on the system.
pub fn count_logical_cores() -> usize {
    unsafe { fu_count_logical_cores() }
}

/// Returns the number of NUMA nodes available on the system.
pub fn count_numa_nodes() -> usize {
    unsafe { fu_count_numa_nodes() }
}

/// Returns the number of distinct thread colocations available.
///
/// A "colocation" represents a group of threads that share the same:
/// - **NUMA memory domain** - threads with fast local memory access
/// - **Quality-of-Service level** - P-cores vs E-cores on heterogeneous CPUs  
/// - **Cache hierarchy** - threads sharing L3 cache
///
/// # Typical Values
///
/// - `1` on most desktop, laptop, or IoT platforms with unified memory
/// - `2-8` on typical dual-socket servers or heterogeneous mobile chips
/// - `4-32` on high-end cloud servers with multiple sockets
pub fn count_colocations() -> usize {
    unsafe { fu_count_colocations() }
}

/// Defines whether the calling thread participates in task execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CallerExclusivity {
    /// The calling thread participates in the workload (spawns N-1 workers)
    Inclusive = 0,
    /// The calling thread only coordinates, doesn't execute tasks (spawns N workers)
    Exclusive = 1,
}

/// Returns the number of distinct Quality-of-Service levels.
pub fn count_quality_levels() -> usize {
    unsafe { fu_count_quality_levels() }
}

/// Returns true if NUMA support was compiled into the library.
pub fn numa_enabled() -> bool {
    unsafe { fu_enabled_numa() != 0 }
}

/// Returns the major version number of the Fork Union library.
pub fn version_major() -> usize {
    unsafe { fu_version_major() as usize }
}

/// Returns the minor version number of the Fork Union library.
pub fn version_minor() -> usize {
    unsafe { fu_version_minor() as usize }
}

/// Returns the patch version number of the Fork Union library.
pub fn version_patch() -> usize {
    unsafe { fu_version_patch() as usize }
}

/// Returns the library version as a tuple of (major, minor, patch).
pub fn version() -> (usize, usize, usize) {
    (version_major(), version_minor(), version_patch())
}

/// Minimalistic, fixed-size thread-pool for blocking scoped parallelism.
///
/// This is a safe Rust wrapper around the precompiled C thread pool implementation.
/// The current thread **participates** in the work, so for `N`-way parallelism the
/// implementation actually spawns **N − 1** background workers and runs the last
/// slice on the caller thread.
///
/// # Thread Safety
///
/// `ThreadPool` is `Send + Sync` and can be safely shared between threads, though
/// operations require a mutable reference to ensure exclusive access during execution.
///
/// # Performance Characteristics
///
/// - Zero dynamic allocations during task execution
/// - Leverages weak memory model for optimal ARM and PowerPC performance  
/// - NUMA-aware thread placement when available
/// - Uses CPU-specific busy-waiting instructions for minimal latency
///
/// # Examples
///
/// Basic usage with simple computations:
///
/// ```rust
/// use fork_union::*;
///
/// // Create a thread pool with 4 threads
/// let mut pool = spawn(4);
///
/// // Execute work on each thread
/// pool.for_threads(|thread_index, colocation_index| {
///     println!("Thread {} on colocation {}", thread_index, colocation_index);
/// });
///
/// // Distribute 1000 tasks across threads
/// pool.for_n(1000, |prong| {
///     // Each task gets a unique index via prong.task_index
///     let result = prong.task_index * prong.task_index;
///     std::hint::black_box(result); // Prevent optimization
/// });
/// ```
///
/// See also helper functions like `for_each_prong_mut` for data processing.
pub struct ThreadPool {
    inner: *mut c_void,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn try_spawn_with_exclusivity(
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(None, threads, exclusivity)
    }

    pub fn try_named_spawn_with_exclusivity(
        name: Option<&str>,
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        if threads == 0 {
            return Err(Error::InvalidParameter);
        }

        unsafe {
            let name_ptr = if let Some(name_str) = name {
                let mut name_buffer = [0u8; 16];
                let name_bytes = name_str.as_bytes();
                let copy_len = core::cmp::min(name_bytes.len(), 15); // Leave space for null terminator
                name_buffer[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
                // name_buffer[copy_len] is already 0 from initialization
                name_buffer.as_ptr() as *const c_char
            } else {
                core::ptr::null()
            };

            let inner = fu_pool_new(name_ptr);
            if inner.is_null() {
                return Err(Error::CreationFailed);
            }

            let success = fu_pool_spawn(inner, threads, exclusivity as c_int);
            if success == 0 {
                fu_pool_delete(inner);
                return Err(Error::SpawnFailed);
            }

            Ok(Self { inner })
        }
    }
    /// Creates a new thread pool with the specified number of threads.
    ///
    /// By default, uses `CallerExclusivity::Inclusive`, meaning the calling thread
    /// participates in work execution. For `N` threads, this spawns `N-1` background
    /// workers plus uses the caller thread.
    ///
    /// # Arguments
    ///
    /// * `threads` - Total number of threads including the caller thread
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// // Create a pool that uses 4 threads total (3 spawned + caller)
    /// let pool = ThreadPool::try_spawn(4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads(), 4);
    /// ```
    pub fn try_spawn(threads: usize) -> Result<Self, Error> {
        Self::try_spawn_with_exclusivity(threads, CallerExclusivity::Inclusive)
    }

    /// Creates a new named thread pool with the specified number of threads.
    ///
    /// The thread pool name can be useful for debugging, profiling, and system monitoring.
    /// On supported platforms, the name may be visible in system tools and thread listings.
    /// Names are truncated to 15 characters (plus null terminator) to fit platform limits.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for the thread pool (up to 15 characters)
    /// * `threads` - Total number of threads including the caller thread
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let pool = ThreadPool::try_named_spawn("worker_pool", 4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads(), 4);
    /// ```
    pub fn try_named_spawn(name: &str, threads: usize) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(Some(name), threads, CallerExclusivity::Inclusive)
    }

    /// Returns the number of threads in the pool.
    pub fn threads(&self) -> usize {
        unsafe { fu_pool_count_threads(self.inner) }
    }

    /// Returns the number of thread colocations in the pool.
    ///
    /// Colocations group threads by NUMA domain, QoS level, and cache hierarchy.
    /// This information is useful for NUMA-aware load balancing and memory allocation.
    pub fn colocations(&self) -> usize {
        unsafe { fu_pool_count_colocations(self.inner) }
    }

    /// Returns the number of threads in a specific colocation.
    ///
    /// This method is useful for NUMA-aware load balancing, allowing you to understand
    /// how many threads are available in each colocation group.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation to query (0-based)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let pool = spawn(8);
    /// let total_colocations = pool.colocations();
    ///
    /// for colocation_index in 0..total_colocations {
    ///     let thread_count = pool.count_threads_in(colocation_index);
    ///     println!("Colocation {} has {} threads", colocation_index, thread_count);
    /// }
    /// ```
    pub fn count_threads_in(&self, colocation_index: usize) -> usize {
        unsafe { fu_pool_count_threads_in(self.inner, colocation_index) }
    }

    /// Converts a global thread index to a local thread index within a colocation.
    ///
    /// This is useful for distributed thread pools where threads are grouped into
    /// colocations (NUMA nodes or QoS levels). The local index can be used for
    /// per-colocation data structures or algorithms.
    ///
    /// # Arguments
    ///
    /// * `global_thread_index` - The global thread index to convert
    /// * `colocation_index` - The colocation to get the local index for
    ///
    /// # Returns
    ///
    /// The local thread index within the specified colocation.
    pub fn locate_thread_in(&self, global_thread_index: usize, colocation_index: usize) -> usize {
        unsafe { fu_pool_locate_thread_in(self.inner, global_thread_index, colocation_index) }
    }

    /// Transitions worker threads to a power-saving sleep state.
    ///
    /// This function places worker threads into a low-power sleep state when no work
    /// is available for extended periods. Threads will periodically check for new work
    /// at the specified interval.
    ///
    /// # Arguments
    ///
    /// * `micros` - Wake-up check interval in microseconds, must be > 0
    ///
    /// # Safety
    ///
    /// This function is **not thread-safe** and should only be called between task batches
    /// when no parallel operations are in progress.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// // Process a batch of work
    /// pool.for_n(1000, |prong| {
    ///     // Do some work...
    ///     std::hint::black_box(prong.task_index * 2);
    /// });
    ///
    /// // Put threads to sleep between batches to save power
    /// // Check for new work every 10 milliseconds
    /// pool.sleep(10_000); // 10,000 microseconds = 10ms
    ///
    /// // Process another batch
    /// pool.for_n(500, |prong| {
    ///     std::hint::black_box(prong.task_index * 3);
    /// });
    /// ```
    pub fn sleep(&mut self, micros: usize) {
        unsafe {
            fu_pool_sleep(self.inner, micros);
        }
    }

    /// Executes a function on each thread of the pool, returning a closure object.
    ///
    /// This operation provides explicit control over broadcast and join phases,
    /// allowing you to start work on threads and then wait for completion separately.
    ///
    /// # Arguments
    ///
    /// * `function` - Closure to execute on each thread, receiving (thread_index, colocation_index)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// {
    ///     let _op = pool.for_threads(|thread_index, colocation_index| {
    ///         println!("Thread {} on colocation {}", thread_index, colocation_index);
    ///         // Simulate some work
    ///         for i in 0..1000 {
    ///             std::hint::black_box(i * thread_index);
    ///         }
    ///     });
    ///     // Work executes when _op is dropped
    /// }
    /// ```
    pub fn for_threads<F>(&mut self, function: F) -> ForThreadsOperation<'_, F>
    where
        F: Fn(usize, usize) + Sync,
    {
        ForThreadsOperation::new(self, function)
    }

    /// Distributes `n` similar duration calls between threads by individual indices.
    ///
    /// Uses static load balancing where each thread gets a predetermined set of tasks.
    /// This is optimal when all tasks have similar execution time.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each task, receiving a `Prong` with task metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_n(1000, |prong| {
    ///     // Simulate computation based on task index
    ///     let result = prong.task_index * prong.task_index;
    ///     std::hint::black_box(result); // Prevent optimization
    /// });
    /// ```
    pub fn for_n<F>(&mut self, n: usize, function: F) -> ForNOperation<'_, F>
    where
        F: Fn(Prong) + Sync,
    {
        ForNOperation {
            pool: self,
            n,
            function,
        }
    }

    /// Executes `n` uneven tasks on all threads, greedily stealing work.
    ///
    /// Uses dynamic load balancing with work-stealing. Threads that finish their
    /// assigned tasks early will steal work from busy threads. This is optimal
    /// when task execution times vary significantly.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each task, receiving a `Prong` with task metadata
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_n_dynamic(100, |prong| {
    ///     // Simulate variable work duration - some tasks take longer
    ///     let iterations = if prong.task_index % 10 == 0 { 10000 } else { 100 };
    ///     for i in 0..iterations {
    ///         std::hint::black_box(prong.task_index * i);
    ///     }
    /// });
    /// ```
    pub fn for_n_dynamic<F>(&mut self, n: usize, function: F) -> ForNDynamicOperation<'_, F>
    where
        F: Fn(Prong) + Sync,
    {
        ForNDynamicOperation {
            pool: self,
            n,
            function,
        }
    }

    /// Distributes `n` similar duration calls between threads in slices.
    ///
    /// Instead of individual task assignment, this method groups tasks into
    /// contiguous slices and assigns each slice to a thread. This reduces
    /// per-task overhead and improves cache locality.
    ///
    /// # Arguments
    ///
    /// * `n` - Total number of tasks to distribute
    /// * `function` - Closure executed for each slice, receiving a `Prong` (with first task index) and slice size
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = spawn(4);
    ///
    /// pool.for_slices(1000, |prong, count| {
    ///     let start_index = prong.task_index;
    ///     
    ///     // Process the slice - each thread gets a contiguous range
    ///     for i in 0..count {
    ///         let global_index = start_index + i;
    ///         let result = global_index * global_index;
    ///         std::hint::black_box(result);
    ///     }
    ///     
    ///     println!("Thread {} processed slice [{}, {})",
    ///              prong.thread_index, start_index, start_index + count);
    /// });
    /// ```
    pub fn for_slices<F>(&mut self, n: usize, function: F) -> ForSlicesOperation<'_, F>
    where
        F: Fn(Prong, usize) + Sync,
    {
        ForSlicesOperation {
            pool: self,
            n,
            function,
        }
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        unsafe {
            fu_pool_terminate(self.inner);
            fu_pool_delete(self.inner);
        }
    }
}

/// Result of a NUMA allocation containing both the allocated pointer and metadata.
#[derive(Debug)]
pub struct AllocationResult {
    ptr: NonNull<u8>,
    allocated_bytes: usize,
    bytes_per_page: usize,
    numa_node: usize,
    // For over-aligned allocations, tracks the unaligned pointer/size for freeing
    overaligned_ptr: Option<NonNull<u8>>,
    overaligned_bytes: Option<usize>,
}

impl AllocationResult {
    /// Returns the allocated memory as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.allocated_bytes) }
    }

    /// Returns the allocated memory as an immutable byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.allocated_bytes) }
    }

    /// Returns the raw pointer to the allocated memory.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Returns the number of bytes actually allocated (may be larger than requested).
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Returns the page size used for this allocation.
    pub fn bytes_per_page(&self) -> usize {
        self.bytes_per_page
    }

    /// Returns the NUMA node this memory was allocated on.
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Converts a typed slice into the allocation's memory space.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `T` has the correct alignment for the allocated memory
    /// - The allocation is large enough to hold the requested number of `T` elements
    /// - The memory is properly initialized before use
    pub unsafe fn as_mut_slice_of<T>(&mut self) -> &mut [T] {
        let element_size = core::mem::size_of::<T>();
        let element_count = self.allocated_bytes / element_size;
        slice::from_raw_parts_mut(self.ptr.as_ptr() as *mut T, element_count)
    }

    /// Converts a typed slice into the allocation's memory space (immutable).
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `T` has the correct alignment for the allocated memory
    /// - The allocation contains valid data of type `T`
    pub unsafe fn as_slice_of<T>(&self) -> &[T] {
        let element_size = core::mem::size_of::<T>();
        let element_count = self.allocated_bytes / element_size;
        slice::from_raw_parts(self.ptr.as_ptr() as *const T, element_count)
    }
}

impl Drop for AllocationResult {
    fn drop(&mut self) {
        unsafe {
            // Use unaligned pointer/size if this was an over-aligned allocation
            let ptr = self.overaligned_ptr.unwrap_or(self.ptr);
            let bytes = self.overaligned_bytes.unwrap_or(self.allocated_bytes);
            fu_free(self.numa_node, ptr.as_ptr() as *mut c_void, bytes);
        }
    }
}

// Safety: AllocationResult can be sent between threads since it owns its memory
unsafe impl Send for AllocationResult {}
// Safety: AllocationResult can be shared between threads with proper synchronization
unsafe impl Sync for AllocationResult {}

/// NUMA-aware memory allocator pinned to a specific NUMA node.
///
/// This allocator provides efficient memory allocation on a specific NUMA node,
/// which is beneficial for performance in multi-socket systems where memory access
/// latency varies based on the physical location of memory relative to the CPU.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
/// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc for NUMA node 0");
/// let allocation = allocator.allocate(1024).expect("Failed to allocate 1024 bytes");
///
/// // Access the allocated memory
/// let memory_slice = allocation.as_slice();
/// assert_eq!(memory_slice.len(), 1024);
/// println!("Allocated {} bytes on NUMA node {}",
///          allocation.allocated_bytes(), allocation.numa_node());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PinnedAllocator {
    numa_node: usize,
}

impl PinnedAllocator {
    /// Creates a new allocator pinned to the specified NUMA node.
    ///
    /// # Arguments
    ///
    /// * `numa_node` - The NUMA node index (0-based) to pin allocations to
    ///
    /// # Errors
    ///
    /// Returns `None` if the NUMA node index is invalid (>= available NUMA nodes).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// // Create allocator for the first NUMA node
    /// let allocator = PinnedAllocator::new(0).expect("NUMA node 0 should be available");
    ///
    /// // Check if a specific NUMA node exists
    /// let numa_count = count_numa_nodes();
    /// if numa_count > 1 {
    ///     let allocator2 = PinnedAllocator::new(1).expect("NUMA node 1 should be available");
    ///     println!("Created allocator for NUMA node: {}", allocator2.numa_node());
    /// }
    /// ```
    pub fn new(numa_node: usize) -> Option<Self> {
        if numa_node >= count_numa_nodes() {
            return None;
        }

        Some(Self { numa_node })
    }

    /// Returns the NUMA node this allocator is pinned to.
    pub fn numa_node(&self) -> usize {
        self.numa_node
    }

    /// Returns the volume of huge pages available on this allocator's NUMA node.
    pub fn volume_huge_pages(&self) -> usize {
        unsafe { fu_volume_huge_pages_in(self.numa_node) }
    }

    /// Returns the volume of any pages (huge or regular) available on this allocator's NUMA node.
    pub fn volume_any_pages(&self) -> usize {
        unsafe { fu_volume_any_pages_in(self.numa_node) }
    }

    /// Allocates memory with at least the requested size on this allocator's NUMA node.
    ///
    /// Returns both the actual allocated size and page size information, which can be
    /// useful for optimizing memory access patterns.
    ///
    /// # Arguments
    ///
    /// * `minimum_bytes` - The minimum number of bytes to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails or if `minimum_bytes` is 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let allocation = allocator.allocate_at_least(1024).expect("Failed to allocate memory");
    ///
    /// println!("Requested 1024 bytes, got {} bytes on {} byte pages",
    ///          allocation.allocated_bytes(), allocation.bytes_per_page());
    ///
    /// // The allocation might be larger than requested due to page alignment
    /// assert!(allocation.allocated_bytes() >= 1024);
    ///
    /// // Access the memory as a byte slice
    /// let memory = allocation.as_slice();
    /// println!("Can access {} bytes of memory", memory.len());
    /// ```
    pub fn allocate_at_least(&self, minimum_bytes: usize) -> Option<AllocationResult> {
        if minimum_bytes == 0 {
            return None;
        }

        let mut allocated_bytes = 0usize;
        let mut bytes_per_page = 0usize;

        unsafe {
            let ptr = fu_allocate_at_least(
                self.numa_node,
                minimum_bytes,
                &mut allocated_bytes as *mut usize,
                &mut bytes_per_page as *mut usize,
            );

            if ptr.is_null() || allocated_bytes == 0 {
                return None;
            }

            Some(AllocationResult {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                allocated_bytes,
                bytes_per_page,
                numa_node: self.numa_node,
                overaligned_ptr: None,
                overaligned_bytes: None,
            })
        }
    }

    /// Allocates exactly the requested number of bytes on this allocator's NUMA node.
    ///
    /// # Arguments
    ///
    /// * `bytes` - The exact number of bytes to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails or if `bytes` is 0.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let allocation = allocator.allocate(1024).expect("Failed to allocate memory");
    /// assert_eq!(allocation.allocated_bytes(), 1024);
    ///
    /// // Write some data to the allocated memory
    /// let mut allocation = allocation; // Make mutable
    /// let memory = allocation.as_mut_slice();
    /// memory[0] = 42;
    /// memory[1023] = 255;
    ///
    /// // Verify the data was written
    /// assert_eq!(memory[0], 42);
    /// assert_eq!(memory[1023], 255);
    /// ```
    pub fn allocate(&self, bytes: usize) -> Option<AllocationResult> {
        if bytes == 0 {
            return None;
        }

        unsafe {
            let ptr = fu_allocate(self.numa_node, bytes);

            if ptr.is_null() {
                return None;
            }

            Some(AllocationResult {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                allocated_bytes: bytes,
                bytes_per_page: 0, // Not provided by fu_allocate
                numa_node: self.numa_node,
                overaligned_ptr: None,
                overaligned_bytes: None,
            })
        }
    }

    /// Allocates memory for a specific number of elements of type T.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of elements to allocate space for
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let mut allocation = allocator.allocate_for::<u64>(100).expect("Failed to allocate");
    ///
    /// // Verify the allocation size first
    /// assert_eq!(allocation.allocated_bytes(), 100 * std::mem::size_of::<u64>());
    ///
    /// // Access as typed slice
    /// let slice = unsafe { allocation.as_mut_slice_of::<u64>() };
    /// slice[0] = 42;
    /// slice[99] = 12345;
    ///
    /// // Read back the values
    /// assert_eq!(slice[0], 42);
    /// assert_eq!(slice[99], 12345);
    /// ```
    pub fn allocate_for<T>(&self, count: usize) -> Option<AllocationResult> {
        let size = core::mem::size_of::<T>();
        let align = core::mem::align_of::<T>();
        let bytes = count.checked_mul(size)?;

        // If alignment is <= default malloc alignment (16 bytes), use simple path
        if align <= 16 {
            return self.allocate(bytes);
        }

        // For over-aligned types (like CacheAligned<T> with 128-byte alignment),
        // we need to over-allocate and manually align the pointer
        let padding = align - 1;
        let total_bytes = bytes.checked_add(padding)?;

        let mut allocation = self.allocate(total_bytes)?;

        // Save unaligned pointer and size for freeing
        let unaligned_ptr = allocation.ptr;
        let unaligned_bytes = allocation.allocated_bytes;

        // Calculate aligned pointer
        let ptr = allocation.as_ptr() as usize;
        let aligned_ptr = (ptr + padding) & !(align - 1);

        // Adjust the allocation to point to the aligned address
        allocation.ptr = unsafe { core::ptr::NonNull::new_unchecked(aligned_ptr as *mut u8) };
        allocation.allocated_bytes = bytes;
        allocation.overaligned_ptr = Some(unaligned_ptr);
        allocation.overaligned_bytes = Some(unaligned_bytes);

        Some(allocation)
    }

    /// Allocates memory for at least the specified number of elements of type T.
    ///
    /// This function may allocate more elements than requested for optimal page alignment.
    ///
    /// # Arguments
    ///
    /// * `min_count` - The minimum number of elements to allocate space for
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).unwrap();
    /// let mut allocation = allocator.allocate_for_at_least::<u32>(1000).expect("Failed to allocate");
    /// let actual_count = allocation.allocated_bytes() / std::mem::size_of::<u32>();
    /// println!("Requested {} u32s, got space for {} u32s", 1000, actual_count);
    ///
    /// // The allocation provides at least the requested number of elements
    /// assert!(actual_count >= 1000);
    ///
    /// // Initialize the allocated memory
    /// let slice = unsafe { allocation.as_mut_slice_of::<u32>() };
    /// for i in 0..1000 {
    ///     slice[i] = i as u32;
    /// }
    ///
    /// // Verify initialization
    /// assert_eq!(slice[0], 0);
    /// assert_eq!(slice[999], 999);
    /// ```
    pub fn allocate_for_at_least<T>(&self, min_count: usize) -> Option<AllocationResult> {
        let min_bytes = min_count.checked_mul(core::mem::size_of::<T>())?;
        self.allocate_at_least(min_bytes)
    }
}

/// Creates an allocator for the first available NUMA node (typically node 0).
///
/// This is a convenience function for systems where NUMA awareness is desired
/// but the specific node doesn't matter.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let allocator = default_numa_allocator().expect("No NUMA nodes available");
/// let allocation = allocator.allocate(1024).expect("Failed to allocate");
///
/// // The default allocator uses NUMA node 0
/// assert_eq!(allocation.numa_node(), 0);
///
/// // For more control, create specific NUMA allocators
/// let numa_count = count_numa_nodes();
/// println!("System has {} NUMA nodes available", numa_count);
///
/// if numa_count > 1 {
///     let allocator_node1 = PinnedAllocator::new(1).expect("NUMA node 1 available");
///     let allocation2 = allocator_node1.allocate(2048).expect("Failed to allocate on node 1");
///     assert_eq!(allocation2.numa_node(), 1);
/// }
/// ```
pub fn default_numa_allocator() -> Option<PinnedAllocator> {
    PinnedAllocator::new(0)
}

/// A Vec-like container that uses NUMA-aware pinned memory allocation.
///
/// `PinnedVec<T>` provides a dynamic array that allocates memory on a specific
/// NUMA node, which should correspond to a `colocation_index` for optimal
/// performance with `ThreadPool`. It automatically manages growth and shrinkage.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// // Create a vector on NUMA node 0
/// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
/// let mut vec = PinnedVec::<u64>::new_in(allocator);
///
/// // Add elements
/// vec.push(42).expect("Failed to push");
/// vec.push(100).expect("Failed to push");
///
/// // Access elements
/// assert_eq!(vec.len(), 2);
/// assert_eq!(vec[0], 42);
/// assert_eq!(vec[1], 100);
///
/// // Iterate over elements
/// for (i, &value) in vec.iter().enumerate() {
///     println!("Element {}: {}", i, value);
/// }
/// ```
#[derive(Debug)]
pub struct PinnedVec<T> {
    allocator: PinnedAllocator,
    allocation: Option<AllocationResult>,
    len: usize,
    capacity: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T> PinnedVec<T> {
    /// Creates a new empty `PinnedVec` using the specified allocator.
    ///
    /// # Arguments
    ///
    /// * `allocator` - The `PinnedAllocator` to use for memory allocation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let vec = PinnedVec::<i32>::new_in(allocator);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 0);
    /// ```
    pub fn new_in(allocator: PinnedAllocator) -> Self {
        Self {
            allocator,
            allocation: None,
            len: 0,
            capacity: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Creates a new `PinnedVec` with the specified capacity using the given allocator.
    ///
    /// # Arguments
    ///
    /// * `allocator` - The `PinnedAllocator` to use for memory allocation
    /// * `capacity` - The initial capacity to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let vec = PinnedVec::<i32>::with_capacity_in(allocator, 100).expect("Failed to create vec");
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 100);
    /// ```
    pub fn with_capacity_in(allocator: PinnedAllocator, capacity: usize) -> Option<Self> {
        let mut vec = Self {
            allocator,
            allocation: None,
            len: 0,
            capacity: 0,
            _phantom: core::marker::PhantomData,
        };

        if capacity > 0 {
            vec.reserve(capacity).ok()?;
        }

        Some(vec)
    }

    /// Returns the number of elements in the vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the NUMA node this vector's memory is allocated on.
    pub fn numa_node(&self) -> usize {
        self.allocator.numa_node()
    }

    /// Reserves capacity for at least `additional` more elements.
    ///
    /// # Arguments
    ///
    /// * `additional` - The number of additional elements to reserve space for
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.reserve(10).expect("Failed to reserve");
    /// assert!(vec.capacity() >= 10);
    /// ```
    pub fn reserve(&mut self, additional: usize) -> Result<(), &'static str> {
        let needed_capacity = self
            .len
            .checked_add(additional)
            .ok_or("Capacity overflow")?;
        if needed_capacity <= self.capacity {
            return Ok(());
        }

        let new_capacity = needed_capacity.max(self.capacity * 2).max(4);
        self.grow_to(new_capacity)
    }

    /// Grows the vector to the specified capacity.
    fn grow_to(&mut self, new_capacity: usize) -> Result<(), &'static str> {
        if new_capacity <= self.capacity {
            return Ok(());
        }

        let new_allocation = self
            .allocator
            .allocate_for::<T>(new_capacity)
            .ok_or("Failed to allocate memory")?;

        if let Some(old_allocation) = self.allocation.take() {
            // Copy existing elements to new allocation
            unsafe {
                let old_ptr = old_allocation.as_ptr() as *const T;
                let new_ptr = new_allocation.as_ptr() as *mut T;
                core::ptr::copy_nonoverlapping(old_ptr, new_ptr, self.len);
            }
        }

        self.allocation = Some(new_allocation);
        self.capacity = new_capacity;
        Ok(())
    }

    /// Appends an element to the back of the vector.
    ///
    /// # Arguments
    ///
    /// * `value` - The element to append
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// assert_eq!(vec.len(), 1);
    /// assert_eq!(vec[0], 42);
    /// ```
    pub fn push(&mut self, value: T) -> Result<(), &'static str> {
        if self.len >= self.capacity {
            self.reserve(1)?;
        }

        unsafe {
            let ptr = self.as_mut_ptr().add(self.len);
            core::ptr::write(ptr, value);
        }
        self.len += 1;
        Ok(())
    }

    /// Removes the last element from the vector and returns it.
    ///
    /// # Returns
    ///
    /// The last element, or `None` if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// assert_eq!(vec.pop(), Some(42));
    /// assert_eq!(vec.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }

        self.len -= 1;
        unsafe {
            let ptr = self.as_mut_ptr().add(self.len);
            Some(core::ptr::read(ptr))
        }
    }

    /// Clears the vector, removing all values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::new_in(allocator);
    /// vec.push(42).expect("Failed to push");
    /// vec.clear();
    /// assert_eq!(vec.len(), 0);
    /// ```
    pub fn clear(&mut self) {
        unsafe {
            let ptr = self.as_mut_ptr();
            for i in 0..self.len {
                core::ptr::drop_in_place(ptr.add(i));
            }
        }
        self.len = 0;
    }

    /// Returns a raw pointer to the vector's buffer.
    pub fn as_ptr(&self) -> *const T {
        match &self.allocation {
            Some(alloc) => alloc.as_ptr() as *const T,
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Returns a mutable raw pointer to the vector's buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        match &self.allocation {
            Some(alloc) => alloc.as_ptr() as *mut T,
            None => core::ptr::NonNull::dangling().as_ptr(),
        }
    }

    /// Returns a synchronization-friendly mutable pointer wrapper.
    ///
    /// The returned pointer can be shared between threads as long as each
    /// thread accesses disjoint indices.
    pub fn sync_ptr(&self) -> SyncMutPtr<T> {
        let ptr = match &self.allocation {
            Some(alloc) => alloc.as_ptr() as *mut T,
            None => core::ptr::NonNull::dangling().as_ptr(),
        };
        SyncMutPtr::new(ptr)
    }

    /// Returns a slice containing the entire vector.
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Returns a mutable slice containing the entire vector.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Returns an iterator over the vector.
    pub fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the vector.
    pub fn iter_mut(&mut self) -> core::slice::IterMut<'_, T> {
        self.as_mut_slice().iter_mut()
    }

    /// Creates a read-only parallel slice view over the vector.
    pub fn par_iter(&self) -> ParallelSlice<'_, T> {
        ParallelSlice::new(self.as_slice())
    }

    /// Creates a mutable parallel slice view over the vector.
    pub fn par_iter_mut(&mut self) -> ParallelSliceMut<'_, T>
    where
        T: Send,
    {
        ParallelSliceMut::new(self.as_mut_slice())
    }

    /// Inserts an element at position `index`, shifting all elements after it to the right.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to insert at
    /// * `element` - The element to insert
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    pub fn insert(&mut self, index: usize, element: T) -> Result<(), &'static str> {
        if index > self.len {
            panic!(
                "insertion index (is {}) should be <= len (is {})",
                index, self.len
            );
        }

        if self.len >= self.capacity {
            self.reserve(1)?;
        }

        unsafe {
            let ptr = self.as_mut_ptr();
            core::ptr::copy(ptr.add(index), ptr.add(index + 1), self.len - index);
            core::ptr::write(ptr.add(index), element);
        }
        self.len += 1;
        Ok(())
    }

    /// Removes and returns the element at position `index`, shifting all elements after it to the left.
    ///
    /// # Arguments
    ///
    /// * `index` - The position to remove from
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len {
            panic!(
                "removal index (is {}) should be < len (is {})",
                index, self.len
            );
        }

        unsafe {
            let ptr = self.as_mut_ptr();
            let result = core::ptr::read(ptr.add(index));
            core::ptr::copy(ptr.add(index + 1), ptr.add(index), self.len - index - 1);
            self.len -= 1;
            result
        }
    }

    /// Extend the vector by cloning elements from a slice.
    ///
    /// # Arguments
    ///
    /// * `other` - The slice to copy elements from
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails when growing the vector.
    pub fn extend_from_slice(&mut self, other: &[T]) -> Result<(), &'static str>
    where
        T: Clone,
    {
        self.reserve(other.len())?;
        for item in other {
            self.push(item.clone())?;
        }
        Ok(())
    }

    /// Returns a reference to an element or subslice depending on the type of index.
    pub fn get<I>(&self, index: I) -> Option<&<I as core::slice::SliceIndex<[T]>>::Output>
    where
        I: core::slice::SliceIndex<[T]>,
    {
        self.as_slice().get(index)
    }

    /// Returns a mutable reference to an element or subslice depending on the type of index.
    pub fn get_mut<I>(
        &mut self,
        index: I,
    ) -> Option<&mut <I as core::slice::SliceIndex<[T]>>::Output>
    where
        I: core::slice::SliceIndex<[T]>,
    {
        self.as_mut_slice().get_mut(index)
    }

    /// Returns a reference to the first element of the vector, or `None` if it is empty.
    pub fn first(&self) -> Option<&T> {
        self.as_slice().first()
    }

    /// Returns a mutable reference to the first element of the vector, or `None` if it is empty.
    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().first_mut()
    }

    /// Returns a reference to the last element of the vector, or `None` if it is empty.
    pub fn last(&self) -> Option<&T> {
        self.as_slice().last()
    }

    /// Returns a mutable reference to the last element of the vector, or `None` if it is empty.
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.as_mut_slice().last_mut()
    }

    /// Swaps two elements in the vector.
    pub fn swap(&mut self, a: usize, b: usize) {
        self.as_mut_slice().swap(a, b)
    }

    /// Reverses the order of elements in the vector, in place.
    pub fn reverse(&mut self) {
        self.as_mut_slice().reverse()
    }

    /// Returns `true` if the vector contains an element with the given value.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        self.as_slice().contains(x)
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            unsafe {
                let ptr = self.as_mut_ptr();
                for i in len..self.len {
                    core::ptr::drop_in_place(ptr.add(i));
                }
            }
            self.len = len;
        }
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    pub fn resize(&mut self, new_len: usize, value: T) -> Result<(), &'static str>
    where
        T: Clone,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len)?;
            while self.len < new_len {
                self.push(value.clone())?;
            }
        } else {
            self.truncate(new_len);
        }
        Ok(())
    }

    /// Resizes the vector in-place so that `len` is equal to `new_len`.
    pub fn resize_with<F>(&mut self, new_len: usize, f: F) -> Result<(), &'static str>
    where
        F: FnMut() -> T,
    {
        if new_len > self.len {
            self.reserve(new_len - self.len)?;
            let mut f = f;
            while self.len < new_len {
                self.push(f())?;
            }
        } else {
            self.truncate(new_len);
        }
        Ok(())
    }

    /// Fills the vector with copies of the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to fill the vector with
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::with_capacity_in(allocator, 5).expect("Failed to create vec");
    /// vec.resize(5, 0).expect("Failed to resize");
    /// vec.fill(42);
    /// assert_eq!(vec.as_slice(), &[42, 42, 42, 42, 42]);
    /// ```
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.as_mut_slice().fill(value);
    }

    /// Fills the vector with values generated by calling a closure repeatedly.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that generates values to fill the vector with
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
    /// let mut vec = PinnedVec::<i32>::with_capacity_in(allocator, 5).expect("Failed to create vec");
    /// vec.resize(5, 0).expect("Failed to resize");
    /// vec.fill_with(|| 42);
    /// assert_eq!(vec.as_slice(), &[42, 42, 42, 42, 42]);
    /// ```
    pub fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        self.as_mut_slice().fill_with(f);
    }
}

impl<T> core::ops::Index<usize> for PinnedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T> core::ops::IndexMut<usize> for PinnedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<T> Drop for PinnedVec<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

unsafe impl<T: Send> Send for PinnedVec<T> {}
unsafe impl<T: Sync> Sync for PinnedVec<T> {}

/// A NUMA-aware distributed vector that manages an array of `PinnedVec`s,
/// each pinned to a specific NUMA node. This structure enables data locality
/// for parallel operations by distributing elements in a round-robin fashion
/// across NUMA nodes, minimizing cross-node memory access penalties.
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let mut pool = ThreadPool::try_spawn(4).expect("Failed to create pool");
/// let mut rr_vec = RoundRobinVec::<i32>::new().expect("Failed to create RoundRobinVec");
///
/// // Fill all vectors across all NUMA nodes with the same value
/// rr_vec.fill(42, &mut pool);
/// ```
pub struct RoundRobinVec<T> {
    colocations: PinnedVec<PinnedVec<T>>,
    total_length: usize,
    total_capacity: usize,
}

impl<T> RoundRobinVec<T> {
    /// Creates a new `RoundRobinVec` with one `PinnedVec` per NUMA node.
    ///
    /// # Errors
    ///
    /// Returns `None` if any of the NUMA node allocators fail to create.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let rr_vec = RoundRobinVec::<i32>::new().expect("Failed to create RoundRobinVec");
    /// assert_eq!(rr_vec.colocations_count(), count_colocations());
    /// ```
    pub fn new() -> Option<Self> {
        let colocations_count = count_colocations();
        if colocations_count == 0 {
            return None;
        }

        // Use the first NUMA node to allocate the container
        let container_allocator = PinnedAllocator::new(0)?;
        let mut colocations = PinnedVec::with_capacity_in(container_allocator, colocations_count)?;

        // Create a PinnedVec for each NUMA node
        for colocation_index in 0..colocations_count {
            let allocator = PinnedAllocator::new(colocation_index)?;
            let vec = PinnedVec::new_in(allocator);
            colocations.push(vec).ok()?;
        }

        let mut total_capacity = 0;
        for i in 0..colocations.len() {
            total_capacity += colocations[i].capacity();
        }

        Some(Self {
            colocations,
            total_length: 0,
            total_capacity,
        })
    }

    /// Creates a new `RoundRobinVec` with pre-allocated capacity on each NUMA node.
    ///
    /// # Arguments
    ///
    /// * `capacity_per_node` - The capacity to allocate on each NUMA node
    ///
    /// # Errors
    ///
    /// Returns `None` if any of the NUMA node allocators fail to create or allocate.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let rr_vec = RoundRobinVec::<i32>::with_capacity_per_colocation(1000)
    ///     .expect("Failed to create RoundRobinVec");
    ///
    /// for i in 0..rr_vec.colocations_count() {
    ///     assert_eq!(rr_vec.capacity_at(i), 1000);
    /// }
    /// ```
    pub fn with_capacity_per_colocation(capacity_per_colocation: usize) -> Option<Self> {
        let colocations_count = count_colocations();
        if colocations_count == 0 {
            return None;
        }

        // Use the first NUMA node to allocate the container
        let container_allocator = PinnedAllocator::new(0)?;
        let mut colocations = PinnedVec::with_capacity_in(container_allocator, colocations_count)?;

        // Create a PinnedVec with capacity for each NUMA node
        for colocation_index in 0..colocations_count {
            let allocator = PinnedAllocator::new(colocation_index)?;
            let vec = PinnedVec::with_capacity_in(allocator, capacity_per_colocation)?;
            colocations.push(vec).ok()?;
        }

        let mut total_capacity = 0;
        for i in 0..colocations.len() {
            total_capacity += colocations[i].capacity();
        }

        Some(Self {
            colocations,
            total_length: 0,
            total_capacity,
        })
    }

    /// Returns the number of colocations (and thus the number of `PinnedVec`s).
    pub fn colocations_count(&self) -> usize {
        self.colocations.len()
    }

    /// Returns the length of the vector at the specified colocation.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation index
    ///
    /// # Returns
    ///
    /// The length of the vector at the specified colocation, or 0 if the node doesn't exist.
    pub fn len_at(&self, colocation_index: usize) -> usize {
        self.colocations[colocation_index].len()
    }

    /// Returns the capacity of the vector at the specified colocation.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation index
    ///
    /// # Returns
    ///
    /// The capacity of the vector at the specified colocation, or 0 if the node doesn't exist.
    pub fn capacity_at(&self, colocation_index: usize) -> usize {
        self.colocations[colocation_index].capacity()
    }

    /// Returns the total length across all NUMA nodes.
    pub fn len(&self) -> usize {
        self.total_length
    }

    /// Returns the total capacity across all NUMA nodes.
    pub fn capacity(&self) -> usize {
        self.total_capacity
    }

    /// Returns `true` if the distributed vector contains no elements.
    pub fn is_empty(&self) -> bool {
        self.total_length == 0
    }

    /// Gets a reference to the `PinnedVec` at the specified colocation.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation index
    ///
    /// # Returns
    ///
    /// A reference to the `PinnedVec` at the specified colocation, or `None` if the node doesn't exist.
    pub fn get_colocation(&self, colocation_index: usize) -> Option<&PinnedVec<T>> {
        self.colocations.get(colocation_index)
    }

    /// Gets a mutable reference to the `PinnedVec` at the specified colocation.
    ///
    /// # Arguments
    ///
    /// * `colocation_index` - The colocation index
    ///
    /// # Returns
    ///
    /// A mutable reference to the `PinnedVec` at the specified colocation, or `None` if the node doesn't exist.
    pub fn get_colocation_mut(&mut self, colocation_index: usize) -> Option<&mut PinnedVec<T>> {
        self.colocations.get_mut(colocation_index)
    }

    /// Accesses an element at a global `index` using round-robin distribution.
    /// The element at global `index` is located at `local_index = index / N`
    /// in the `PinnedVec` on `numa_node = index % N`, where `N` is the number of NUMA nodes.
    ///
    /// # Arguments
    ///
    /// * `index` - The global round-robin index of the element.
    ///
    /// # Returns
    ///
    /// A reference to the element, or `None` if the index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut rr_vec = RoundRobinVec::<i32>::new().expect("Failed to create RoundRobinVec");
    /// // Add some elements...
    /// if let Some(element) = rr_vec.get(5) {
    ///     println!("Element at index 5: {}", element);
    /// }
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        if self.colocations.is_empty() {
            return None;
        }

        let colocations_count = self.colocations.len();
        let colocation_index = index % colocations_count;
        let local_index = index / colocations_count;

        self.colocations.get(colocation_index)?.get(local_index)
    }

    /// Mutably accesses an element at a global `index` using round-robin distribution.
    /// The element at global `index` is located at `local_index = index / N`
    /// in the `PinnedVec` on `numa_node = index % N`, where `N` is the number of NUMA nodes.
    ///
    /// # Arguments
    ///
    /// * `index` - The global round-robin index of the element.
    ///
    /// # Returns
    ///
    /// A mutable reference to the element, or `None` if the index is out of bounds.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if self.colocations.is_empty() {
            return None;
        }

        let colocations_count = self.colocations.len();
        let colocation_index = index % colocations_count;
        let local_index = index / colocations_count;

        self.colocations
            .get_mut(colocation_index)?
            .get_mut(local_index)
    }

    /// Appends an element to the distributed vector, using round-robin distribution
    /// to select the target NUMA node based on the current total length.
    ///
    /// # Arguments
    ///
    /// * `value` - The element to add.
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails on the target NUMA node.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut rr_vec = RoundRobinVec::<i32>::new().expect("Failed to create RoundRobinVec");
    /// rr_vec.push(42).expect("Failed to push");
    /// assert_eq!(rr_vec.len(), 1);
    /// ```
    pub fn push(&mut self, value: T) -> Result<(), &'static str> {
        if self.colocations.is_empty() {
            return Err("No NUMA nodes available");
        }

        // Use round-robin distribution based on current total length
        let target_colocation = self.total_length % self.colocations.len();
        let result = self.colocations[target_colocation].push(value);

        if result.is_ok() {
            self.total_length += 1;
        }

        result
    }

    /// Creates a parallel read-only iterator over all elements in round-robin order.
    pub fn par_iter(&self) -> ParallelRoundRobin<'_, T>
    where
        T: Sync,
    {
        ParallelRoundRobin::new(self)
    }

    /// Creates a parallel mutable iterator over all elements in round-robin order.
    pub fn par_iter_mut(&mut self) -> ParallelRoundRobinMut<'_, T>
    where
        T: Send,
    {
        ParallelRoundRobinMut::new(self)
    }

    /// Executes a closure in parallel on each element, preserving round-robin distribution.
    pub fn par_for_each<S, F>(&self, pool: &mut ThreadPool, schedule: S, function: F)
    where
        T: Sync,
        S: ParallelSchedule,
        F: Fn(&T, Prong) + Sync,
    {
        self.par_iter().drive(pool, schedule, &function);
    }

    /// Executes a mutable closure in parallel on each element.
    pub fn par_for_each_mut<S, F>(&mut self, pool: &mut ThreadPool, schedule: S, function: F)
    where
        T: Send,
        S: ParallelSchedule,
        F: Fn(&mut T, Prong) + Sync,
    {
        self.par_iter_mut().drive(pool, schedule, &function);
    }

    /// Removes and returns the last element from the distributed vector.
    /// The element is popped from the NUMA node it was last pushed to, maintaining
    /// round-robin balance.
    ///
    /// # Returns
    ///
    /// The last element, or `None` if the vector is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut rr_vec = RoundRobinVec::<i32>::new().expect("Failed to create RoundRobinVec");
    /// rr_vec.push(42).expect("Failed to push");
    /// assert_eq!(rr_vec.pop(), Some(42));
    /// assert_eq!(rr_vec.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        if self.total_length == 0 {
            return None;
        }

        // Pop from the last inserted position (reverse round-robin)
        let target_colocation = (self.total_length - 1) % self.colocations.len();
        let result = self.colocations[target_colocation].pop();

        if result.is_some() {
            self.total_length -= 1;
        }

        result
    }

    /// Converts a local index within a specific NUMA node to the global round-robin index.
    ///
    /// This is useful when you have a reference to an element within a specific `PinnedVec`
    /// and need to determine its global index in the round-robin distribution.
    ///
    /// # Arguments
    ///
    /// * `numa_node` - The NUMA node index (0 to numa_count()-1)
    /// * `local_index` - The local index within that NUMA node's `PinnedVec`
    ///
    /// # Returns
    ///
    /// The global index where this element would be accessed via `get(global_index)`.
    pub fn local_to_global_index(&self, colocation_index: usize, local_index: usize) -> usize {
        local_index * self.colocations_count() + colocation_index
    }

    /// Converts a global round-robin index to the NUMA node and local index.
    ///
    /// This is the inverse of `local_to_global_index`.
    ///
    /// # Arguments
    ///
    /// * `global_index` - The global index in the round-robin distribution
    ///
    /// # Returns
    ///
    /// A tuple of (colocation_index, local_index) where the element is stored.
    pub fn global_to_local_index(&self, global_index: usize) -> (usize, usize) {
        let colocations_count = self.colocations_count();
        let colocation_index = global_index % colocations_count;
        let local_index = global_index / colocations_count;
        (colocation_index, local_index)
    }

    /// Fills all vectors across all NUMA nodes with copies of the given value,
    /// using the thread pool for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to fill all vectors with
    /// * `pool` - The thread pool to use for parallel execution
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).expect("Failed to create pool");
    /// let mut rr_vec = RoundRobinVec::<i32>::with_capacity_per_colocation(1000)
    ///     .expect("Failed to create RoundRobinVec");
    ///
    /// // Resize all vectors to have some elements
    /// for i in 0..rr_vec.colocations_count() {
    ///     rr_vec.get_colocation_mut(i).unwrap().resize(100, 0).expect("Failed to resize");
    /// }
    ///
    /// // Fill all vectors with the value 42
    /// rr_vec.fill(42, &mut pool);
    /// ```
    pub fn fill(&mut self, value: T, pool: &mut ThreadPool)
    where
        T: Clone + Send + Sync,
    {
        let colocations_count = self.colocations_count();
        let safe_ptr = SafePtr(self.colocations.as_mut_ptr());
        let pool_ptr = SafePtr(pool as *const ThreadPool as *mut ThreadPool);

        pool.for_threads(move |thread_index, colocation_index| {
            if colocation_index >= colocations_count {
                return;
            }

            let node_vec = safe_ptr.get_mut_at(colocation_index);
            let pool = pool_ptr.get_mut();

            let threads_in_colocation = pool.count_threads_in(colocation_index);
            let thread_local_index = pool.locate_thread_in(thread_index, colocation_index);
            let split = IndexedSplit::new(node_vec.len(), threads_in_colocation);
            let range = split.get(thread_local_index);

            // Fill the assigned range of this thread
            for idx in range {
                if let Some(element) = node_vec.get_mut(idx) {
                    *element = value.clone();
                }
            }
        });
    }

    /// Fills all vectors across all NUMA nodes with values generated by calling
    /// a closure repeatedly, using the thread pool for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that generates values to fill the vectors with
    /// * `pool` - The thread pool to use for parallel execution
    ///
    /// # Examples
    ///
    /// ```rust
    /// use fork_union::*;
    ///
    /// let mut pool = ThreadPool::try_spawn(4).expect("Failed to create pool");
    /// let mut rr_vec = RoundRobinVec::<i32>::with_capacity_per_colocation(1000)
    ///     .expect("Failed to create RoundRobinVec");
    ///
    /// // Resize all vectors to have some elements
    /// for i in 0..rr_vec.colocations_count() {
    ///     rr_vec.get_colocation_mut(i).unwrap().resize(100, 0).expect("Failed to resize");
    /// }
    ///
    /// // Fill all vectors with random values
    /// rr_vec.fill_with(|| rand::random::<i32>(), &mut pool);
    /// ```
    pub fn fill_with<F>(&mut self, mut f: F, pool: &mut ThreadPool)
    where
        F: FnMut() -> T + Send + Sync,
        T: Send + Sync,
    {
        let colocations_count = self.colocations_count();
        let safe_ptr = SafePtr(self.colocations.as_mut_ptr());
        let f_ptr = SafePtr(&mut f as *mut F);
        let pool_ptr = SafePtr(pool as *const ThreadPool as *mut ThreadPool);

        pool.for_threads(move |thread_index, colocation_index| {
            if colocation_index >= colocations_count {
                return;
            }

            let node_vec = safe_ptr.get_mut_at(colocation_index);
            let f_ref = f_ptr.get_mut();
            let pool = pool_ptr.get_mut();

            let threads_in_colocation = pool.count_threads_in(colocation_index);
            let thread_local_index = pool.locate_thread_in(thread_index, colocation_index);
            let split = IndexedSplit::new(node_vec.len(), threads_in_colocation);
            let range = split.get(thread_local_index);

            // Fill the assigned range of this thread
            for idx in range {
                if let Some(element) = node_vec.get_mut(idx) {
                    *element = f_ref();
                }
            }
        });
    }

    /// Clears all vectors across all NUMA nodes, using the thread pool for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `pool` - The thread pool to use for parallel execution
    pub fn clear(&mut self, pool: &mut ThreadPool) {
        let colocations_count = self.colocations_count();
        let safe_ptr = SafePtr(self.colocations.as_mut_ptr());
        let pool_ptr = SafePtr(pool as *const ThreadPool as *mut ThreadPool);

        pool.for_threads(move |thread_index, colocation_index| {
            if colocation_index >= colocations_count {
                return;
            }

            let node_vec = safe_ptr.get_mut_at(colocation_index);
            let pool = pool_ptr.get_mut();

            let threads_in_colocation = pool.count_threads_in(colocation_index);
            let thread_local_index = pool.locate_thread_in(thread_index, colocation_index);
            let split = IndexedSplit::new(node_vec.len(), threads_in_colocation);
            let range = split.get(thread_local_index);

            // Drop elements in the assigned range
            unsafe {
                let ptr = node_vec.as_mut_ptr();
                for idx in range {
                    core::ptr::drop_in_place(ptr.add(idx));
                }
            }
        });

        // Reset lengths of individual vectors after parallel dropping
        for i in 0..self.colocations.len() {
            self.colocations[i].len = 0;
        }
        self.total_length = 0;
    }

    /// Resizes all vectors across all NUMA nodes to the specified length,
    /// using the thread pool for parallel execution.
    ///
    /// # Arguments
    ///
    /// * `new_len` - The new length for all vectors
    /// * `value` - The value to fill new elements with
    /// * `pool` - The thread pool to use for parallel execution
    ///
    /// # Errors
    ///
    /// Returns an error if any vector fails to resize.
    pub fn resize(
        &mut self,
        new_len: usize,
        value: T,
        pool: &mut ThreadPool,
    ) -> Result<(), &'static str>
    where
        T: Clone + Send + Sync,
    {
        let colocations_count = self.colocations_count();
        if colocations_count == 0 {
            return Err("No NUMA nodes available");
        }

        // Calculate how many elements each NUMA node should have
        let elements_per_node = new_len / colocations_count;
        let extra_elements = new_len % colocations_count;

        // Helper to calculate target length for a colocation
        let node_len = |col_idx: usize| -> usize {
            if col_idx < extra_elements {
                elements_per_node + 1
            } else {
                elements_per_node
            }
        };

        // Step 1: Centrally handle reallocation for each NUMA node
        for i in 0..colocations_count {
            let target_len = node_len(i);
            let current_len = self.colocations[i].len();
            if target_len > current_len {
                self.colocations[i].reserve(target_len - current_len)?;
            }
        }

        // Step 2: Parallel construction/destruction of elements using IndexedSplit
        let safe_ptr = SafePtr(self.colocations.as_mut_ptr());
        let pool_ptr = SafePtr(pool as *const ThreadPool as *mut ThreadPool);

        pool.for_threads(move |thread_index, colocation_index| {
            if colocation_index >= colocations_count {
                return;
            }

            let node_vec = safe_ptr.get_mut_at(colocation_index);
            let pool = pool_ptr.get_mut();
            let target_len = node_len(colocation_index);
            let current_len = node_vec.len();
            if target_len == current_len {
                return;
            }

            let threads_in_colocation = pool.count_threads_in(colocation_index);
            let thread_local_index = pool.locate_thread_in(thread_index, colocation_index);

            if target_len > current_len {
                // Growing: construct new elements in parallel
                let new_elements = target_len - current_len;
                let split = IndexedSplit::new(new_elements, threads_in_colocation);
                let range = split.get(thread_local_index);

                unsafe {
                    let ptr = node_vec.as_mut_ptr();
                    for i in range {
                        core::ptr::write(ptr.add(current_len + i), value.clone());
                    }
                }
            } else {
                // Shrinking: drop elements in parallel
                let elements_to_drop = current_len - target_len;
                let split = IndexedSplit::new(elements_to_drop, threads_in_colocation);
                let range = split.get(thread_local_index);

                unsafe {
                    let ptr = node_vec.as_mut_ptr();
                    for i in range {
                        core::ptr::drop_in_place(ptr.add(target_len + i));
                    }
                }
            }
        });

        // Step 3: Update lengths after parallel operations
        for i in 0..colocations_count {
            self.colocations[i].len = node_len(i);
        }

        self.total_length = new_len;
        self.total_capacity = self.capacity();
        Ok(())
    }
}

/// A thread-safe wrapper for raw pointers used in parallel operations.
///
/// # Safety
/// This wrapper is only safe when used with NUMA-aware thread pools where
/// each thread accesses different memory locations (different NUMA nodes).
pub struct SafePtr<T>(*mut T);
unsafe impl<T> Send for SafePtr<T> {}
unsafe impl<T> Sync for SafePtr<T> {}

impl<T> SafePtr<T> {
    /// Creates a new SafePtr from a raw pointer.
    pub fn new(ptr: *mut T) -> Self {
        SafePtr(ptr)
    }

    /// Accesses the element at the given index.
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut_at(&self, index: usize) -> &mut T {
        unsafe { &mut *self.0.add(index) }
    }

    /// Accesses the element.
    #[allow(clippy::mut_from_ref)]
    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.0 }
    }
}

unsafe impl<T: Send> Send for RoundRobinVec<T> {}
unsafe impl<T: Sync> Sync for RoundRobinVec<T> {}

/// A thread-safe wrapper around raw pointers for sharing read-only data across threads.
///
/// This type is designed for scenarios where you need to share immutable data
/// across async tasks or threads, particularly when the standard borrowing rules
/// would prevent such sharing. The caller is responsible for ensuring that:
/// - The pointed-to data remains valid for the lifetime of use
/// - The data is not modified while being accessed through `SyncConstPtr`
///
/// # Safety
///
/// This type is marked as `Send + Sync` but requires careful usage:
/// - Only use with data that won't be modified during the lifetime of the pointer
/// - Ensure the pointed-to data outlives all uses of the `SyncConstPtr`
/// - The `get` method is unsafe and requires the caller to ensure bounds checking
///
/// # Examples
///
/// ```rust
/// use fork_union::*;
///
/// let data = vec![1, 2, 3, 4, 5];
/// let sync_ptr = SyncConstPtr::new(data.as_ptr());
///
/// // Safe to use in async contexts
/// let value = unsafe { sync_ptr.get(0) };
/// assert_eq!(*value, 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SyncConstPtr<T> {
    ptr: *const T,
}

impl<T> SyncConstPtr<T> {
    /// Creates a new `SyncConstPtr` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The pointer is valid for the intended usage duration
    /// - The pointed-to data will not be modified during use
    /// - The pointer is properly aligned for type `T`
    pub fn new(ptr: *const T) -> Self {
        Self { ptr }
    }

    /// Gets a reference to the element at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The index is within bounds of the allocated data
    /// - The data at the index is properly initialized
    /// - The data remains valid for the lifetime of the returned reference
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the element to access
    ///
    /// # Returns
    ///
    /// A reference to the element at the given index.
    pub unsafe fn get(&self, index: usize) -> &T {
        &*self.ptr.add(index)
    }

    /// Returns the raw pointer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
}

unsafe impl<T> Send for SyncConstPtr<T> {}
unsafe impl<T> Sync for SyncConstPtr<T> {}

#[derive(Clone, Copy)]
pub struct SyncMutPtr<T> {
    ptr: *mut T,
    _marker: PhantomData<T>,
}

impl<T> SyncMutPtr<T> {
    pub const fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Returns a mutable pointer to the element at the given index.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The index is within the bounds of the original allocation
    /// - No overlapping mutable access occurs from multiple threads
    /// - Each thread accesses disjoint indices when used concurrently
    /// - The pointer remains valid for the duration of access
    pub unsafe fn get(&self, index: usize) -> *mut T {
        self.ptr.add(index)
    }

    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }
}

unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

/// Scheduler that uses static chunk assignment.
#[derive(Clone, Copy, Debug)]
pub struct StaticScheduler;

/// Scheduler that enables dynamic work stealing.
#[derive(Clone, Copy, Debug)]
pub struct DynamicScheduler;

pub trait ParallelSchedule: Copy {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F)
    where
        F: Fn(Prong) + Sync;

    fn dispatch_slices<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F)
    where
        F: Fn(Prong, usize) + Sync,
    {
        self.dispatch(pool, tasks, move |prong| {
            function(prong, 1);
        });
    }
}

impl ParallelSchedule for StaticScheduler {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F)
    where
        F: Fn(Prong) + Sync,
    {
        if tasks == 0 {
            return;
        }

        let function_ptr = SyncConstPtr::new(&function as *const F);
        let _operation = pool.for_n(tasks, move |prong| {
            let func = unsafe { &*function_ptr.as_ptr() };
            func(prong);
        });
    }

    fn dispatch_slices<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F)
    where
        F: Fn(Prong, usize) + Sync,
    {
        if tasks == 0 {
            return;
        }

        let function_ptr = SyncConstPtr::new(&function as *const F);
        let _operation = pool.for_slices(tasks, move |prong, count| {
            let func = unsafe { &*function_ptr.as_ptr() };
            func(prong, count);
        });
    }
}

impl ParallelSchedule for DynamicScheduler {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F)
    where
        F: Fn(Prong) + Sync,
    {
        if tasks == 0 {
            return;
        }

        let function_ptr = SyncConstPtr::new(&function as *const F);
        let _operation = pool.for_n_dynamic(tasks, move |prong| {
            let func = unsafe { &*function_ptr.as_ptr() };
            func(prong);
        });
    }
}

pub trait ParallelIterator: Sized {
    type Item;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync;

    fn drive_static<F>(self, pool: &mut ThreadPool, consumer: F)
    where
        F: Fn(Self::Item, Prong) + Sync,
    {
        self.drive(pool, StaticScheduler, &consumer);
    }

    fn drive_dynamic<F>(self, pool: &mut ThreadPool, consumer: F)
    where
        F: Fn(Self::Item, Prong) + Sync,
    {
        self.drive(pool, DynamicScheduler, &consumer);
    }

    fn map<M, U>(self, mapper: M) -> Map<Self, M>
    where
        M: Fn(Self::Item) -> U + Sync,
    {
        Map { base: self, mapper }
    }

    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        P: Fn(&Self::Item) -> bool + Sync,
    {
        Filter {
            base: self,
            predicate,
        }
    }
}

pub trait ParallelIteratorExt: ParallelIterator + Sized {
    fn with_pool<'pool>(
        self,
        pool: &'pool mut ThreadPool,
    ) -> ParallelRunner<'pool, Self, StaticScheduler> {
        ParallelRunner {
            pool,
            iterator: self,
            schedule: StaticScheduler,
        }
    }

    fn with_schedule<'pool, S>(
        self,
        pool: &'pool mut ThreadPool,
        schedule: S,
    ) -> ParallelRunner<'pool, Self, S>
    where
        S: ParallelSchedule,
    {
        ParallelRunner {
            pool,
            iterator: self,
            schedule,
        }
    }
}

impl<I: ParallelIterator> ParallelIteratorExt for I {}

pub struct ParallelRunner<'pool, I, S> {
    pool: &'pool mut ThreadPool,
    iterator: I,
    schedule: S,
}

impl<'pool, I, S> ParallelRunner<'pool, I, S>
where
    I: ParallelIterator,
    S: ParallelSchedule,
{
    pub fn for_each<F>(self, function: F)
    where
        F: Fn(I::Item) + Sync,
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;
        let function_ptr = SyncConstPtr::new(&function as *const F);
        iterator.drive(pool, schedule, &move |item, _| {
            let func = unsafe { &*function_ptr.as_ptr() };
            func(item);
        });
    }

    pub fn for_each_with_prong<F>(self, function: F)
    where
        F: Fn(I::Item, Prong) + Sync,
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;
        let function_ptr = SyncConstPtr::new(&function as *const F);
        iterator.drive(pool, schedule, &move |item, prong| {
            let func = unsafe { &*function_ptr.as_ptr() };
            func(item, prong);
        });
    }

    pub fn fold_with_scratch<T, F>(self, scratch: &mut [T], fold: F)
    where
        T: Send,
        F: Fn(&mut T, I::Item, Prong) + Sync,
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;
        fold_with_scratch(pool, iterator, schedule, scratch, fold);
    }

    pub fn with_schedule<S2>(self, schedule: S2) -> ParallelRunner<'pool, I, S2>
    where
        S2: ParallelSchedule,
    {
        let ParallelRunner { pool, iterator, .. } = self;
        ParallelRunner {
            pool,
            iterator,
            schedule,
        }
    }
}

pub trait IntoParallelIterator {
    type Item;
    type Iter: ParallelIterator<Item = Self::Item>;

    fn into_par_iter(self) -> Self::Iter;
}

impl<'a, T> IntoParallelIterator for &'a [T]
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = ParallelSlice<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        ParallelSlice::new(self)
    }
}

impl<'a, T> IntoParallelIterator for &'a mut [T]
where
    T: Send,
{
    type Item = &'a mut T;
    type Iter = ParallelSliceMut<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        ParallelSliceMut::new(self)
    }
}

impl IntoParallelIterator for core::ops::Range<usize> {
    type Item = usize;
    type Iter = ParallelRange;

    fn into_par_iter(self) -> Self::Iter {
        ParallelRange::new(self)
    }
}

impl<'a, T> IntoParallelIterator for &'a RoundRobinVec<T>
where
    T: Sync,
{
    type Item = &'a T;
    type Iter = ParallelRoundRobin<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        ParallelRoundRobin::new(self)
    }
}

impl<'a, T> IntoParallelIterator for &'a mut RoundRobinVec<T>
where
    T: Send,
{
    type Item = &'a mut T;
    type Iter = ParallelRoundRobinMut<'a, T>;

    fn into_par_iter(self) -> Self::Iter {
        ParallelRoundRobinMut::new(self)
    }
}

pub struct Map<I, M> {
    base: I,
    mapper: M,
}

impl<I, M, U> ParallelIterator for Map<I, M>
where
    I: ParallelIterator,
    M: Fn(I::Item) -> U + Sync,
{
    type Item = U;

    fn len(&self) -> usize {
        self.base.len()
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        let Map { base, mapper } = self;
        let mapper_ptr = SyncConstPtr::new(&mapper as *const M);
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        let mapped = move |item: I::Item, prong: Prong| {
            let mp = unsafe { &*mapper_ptr.as_ptr() };
            let next = mp(item);
            let consumer_ref = unsafe { &*consumer_ptr.as_ptr() };
            consumer_ref(next, prong);
        };

        base.drive(pool, schedule, &mapped);
    }
}

pub struct Filter<I, P> {
    base: I,
    predicate: P,
}

impl<I, P> ParallelIterator for Filter<I, P>
where
    I: ParallelIterator,
    P: Fn(&I::Item) -> bool + Sync,
{
    type Item = I::Item;

    fn len(&self) -> usize {
        self.base.len()
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        let Filter { base, predicate } = self;
        let predicate_ptr = SyncConstPtr::new(&predicate as *const P);
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        let filtered = move |item: I::Item, prong: Prong| {
            let pred = unsafe { &*predicate_ptr.as_ptr() };
            if pred(&item) {
                let consumer_ref = unsafe { &*consumer_ptr.as_ptr() };
                consumer_ref(item, prong);
            }
        };

        base.drive(pool, schedule, &filtered);
    }
}

#[derive(Clone, Copy)]
pub struct ParallelSlice<'a, T> {
    data: &'a [T],
}

impl<'a, T> ParallelSlice<'a, T> {
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    pub fn for_each_static<F>(self, pool: &mut ThreadPool, function: F)
    where
        T: Sync,
        F: Fn(&'a T, Prong) + Sync,
    {
        self.drive_static(pool, function);
    }

    pub fn for_each_dynamic<F>(self, pool: &mut ThreadPool, function: F)
    where
        T: Sync,
        F: Fn(&'a T, Prong) + Sync,
    {
        self.drive_dynamic(pool, function);
    }

    pub fn zip<'b, U>(self, other: ParallelSlice<'b, U>) -> ParallelSliceZip<'a, 'b, T, U> {
        ParallelSliceZip {
            left: self,
            right: other,
        }
    }
}

impl<'a, T> ParallelIterator for ParallelSlice<'a, T>
where
    T: Sync,
{
    type Item = &'a T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        if self.data.is_empty() {
            return;
        }

        let slice = self.data;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, slice.len(), move |prong| {
            let item = unsafe { slice.get_unchecked(prong.task_index) };
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func(item, prong);
        });
    }
}

pub struct ParallelSliceZip<'a, 'b, T, U> {
    left: ParallelSlice<'a, T>,
    right: ParallelSlice<'b, U>,
}

impl<'a, 'b, T, U> ParallelIterator for ParallelSliceZip<'a, 'b, T, U>
where
    T: Sync,
    U: Sync,
{
    type Item = (&'a T, &'b U);

    fn len(&self) -> usize {
        let len = self.left.data.len();
        debug_assert_eq!(len, self.right.data.len());
        len
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        let len = self.left.data.len();
        assert_eq!(len, self.right.data.len(), "zip requires equal lengths");
        if len == 0 {
            return;
        }

        let left = self.left.data;
        let right = self.right.data;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, len, move |prong| {
            let lhs = unsafe { left.get_unchecked(prong.task_index) };
            let rhs = unsafe { right.get_unchecked(prong.task_index) };
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func((lhs, rhs), prong);
        });
    }
}

pub struct ParallelSliceMut<'a, T> {
    ptr: SyncMutPtr<T>,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> ParallelSliceMut<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self {
            ptr: SyncMutPtr::new(data.as_mut_ptr()),
            len: data.len(),
            _marker: PhantomData,
        }
    }

    pub fn for_each_static<F>(self, pool: &mut ThreadPool, function: F)
    where
        T: Send,
        F: Fn(&'a mut T, Prong) + Sync,
    {
        self.drive_static(pool, function);
    }

    pub fn for_each_dynamic<F>(self, pool: &mut ThreadPool, function: F)
    where
        T: Send,
        F: Fn(&'a mut T, Prong) + Sync,
    {
        self.drive_dynamic(pool, function);
    }
}

impl<'a, T> ParallelIterator for ParallelSliceMut<'a, T>
where
    T: Send,
{
    type Item = &'a mut T;

    fn len(&self) -> usize {
        self.len
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        if self.len == 0 {
            return;
        }

        let ptr = self.ptr;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, self.len, move |prong| {
            let raw = unsafe { ptr.get(prong.task_index) };
            let item = unsafe { &mut *raw };
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func(item, prong);
        });
    }
}

#[derive(Clone)]
pub struct ParallelRange {
    range: core::ops::Range<usize>,
}

impl ParallelRange {
    pub fn new(range: core::ops::Range<usize>) -> Self {
        Self { range }
    }
}

impl ParallelIterator for ParallelRange {
    type Item = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        let len = self.range.len();
        if len == 0 {
            return;
        }

        let start = self.range.start;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, len, move |mut prong| {
            let index = start + prong.task_index;
            prong.task_index = index;
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func(index, prong);
        });
    }
}

pub struct ParallelExactIter<T, I> {
    len: usize,
    indexer: I,
    _marker: PhantomData<T>,
}

impl<T, I> ParallelExactIter<T, I>
where
    I: Fn(usize) -> T + Sync,
{
    pub fn new(len: usize, indexer: I) -> Self {
        Self {
            len,
            indexer,
            _marker: PhantomData,
        }
    }
}

impl<T, I> ParallelIterator for ParallelExactIter<T, I>
where
    T: Send,
    I: Fn(usize) -> T + Sync,
{
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        let ParallelExactIter { len, indexer, .. } = self;
        if len == 0 {
            return;
        }

        let indexer_ptr = SyncConstPtr::new(&indexer as *const I);
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch_slices(pool, len, move |mut prong, count| {
            let mut current = prong.task_index;
            let idx_fn = unsafe { &*indexer_ptr.as_ptr() };
            let consumer_ref = unsafe { &*consumer_ptr.as_ptr() };
            for _ in 0..count {
                prong.task_index = current;
                let value = idx_fn(current);
                consumer_ref(value, prong);
                current += 1;
            }
        });
    }
}

pub struct ParallelRoundRobin<'a, T> {
    colocations_ptr: SyncConstPtr<PinnedVec<T>>,
    colocations_len: usize,
    total_len: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T> ParallelRoundRobin<'a, T> {
    fn new(vec: &'a RoundRobinVec<T>) -> Self {
        let slice = vec.colocations.as_slice();
        Self {
            colocations_ptr: SyncConstPtr::new(slice.as_ptr()),
            colocations_len: slice.len(),
            total_len: vec.total_length,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> ParallelIterator for ParallelRoundRobin<'a, T>
where
    T: Sync,
{
    type Item = &'a T;

    fn len(&self) -> usize {
        self.total_len
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        if self.total_len == 0 || self.colocations_len == 0 {
            return;
        }

        let colocations_ptr = self.colocations_ptr;
        let colocations_len = self.colocations_len;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, self.total_len, move |prong| {
            let index = prong.task_index;
            let colocation_index = index % colocations_len;
            let local_index = index / colocations_len;
            let base = colocations_ptr.as_ptr();
            let colocation = unsafe { &*base.add(colocation_index) };
            let slice = colocation.as_slice();
            let item = unsafe { slice.get_unchecked(local_index) };
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func(item, prong);
        });
    }
}

pub struct ParallelRoundRobinMut<'a, T> {
    colocations_ptr: SyncConstPtr<PinnedVec<T>>,
    colocations_len: usize,
    total_len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> ParallelRoundRobinMut<'a, T> {
    fn new(vec: &'a mut RoundRobinVec<T>) -> Self {
        let slice = vec.colocations.as_slice();
        Self {
            colocations_ptr: SyncConstPtr::new(slice.as_ptr()),
            colocations_len: slice.len(),
            total_len: vec.total_length,
            _marker: PhantomData,
        }
    }
}

impl<'a, T> ParallelIterator for ParallelRoundRobinMut<'a, T>
where
    T: Send,
{
    type Item = &'a mut T;

    fn len(&self) -> usize {
        self.total_len
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F)
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, Prong) + Sync,
    {
        if self.total_len == 0 || self.colocations_len == 0 {
            return;
        }

        let colocations_ptr = self.colocations_ptr;
        let colocations_len = self.colocations_len;
        let consumer_ptr = SyncConstPtr::new(consumer as *const F);
        schedule.dispatch(pool, self.total_len, move |prong| {
            let index = prong.task_index;
            let colocation_index = index % colocations_len;
            let local_index = index / colocations_len;
            let base = colocations_ptr.as_ptr();
            let colocation = unsafe { &*base.add(colocation_index) };
            let base_ptr = colocation.sync_ptr();
            let raw = unsafe { base_ptr.get(local_index) };
            let item = unsafe { &mut *raw };
            let func = unsafe { &*consumer_ptr.as_ptr() };
            func(item, prong);
        });
    }
}

pub fn fold_with_scratch<I, S, T, F>(
    pool: &mut ThreadPool,
    iterator: I,
    schedule: S,
    scratch: &mut [T],
    fold: F,
) where
    I: ParallelIterator,
    S: ParallelSchedule,
    T: Send,
    F: Fn(&mut T, I::Item, Prong) + Sync,
{
    let scratch_len = scratch.len();
    assert!(
        scratch_len >= pool.threads(),
        "scratch space must cover all threads"
    );
    let scratch_ptr = SyncMutPtr::new(scratch.as_mut_ptr());
    iterator.drive(pool, schedule, &move |item, prong| {
        debug_assert!(prong.thread_index < scratch_len);
        let slot = unsafe { &mut *scratch_ptr.get(prong.thread_index) };
        fold(slot, item, prong);
    });
}

pub mod prelude {
    pub use super::{
        DynamicScheduler, IntoParallelIterator, ParallelIterator, ParallelIteratorExt,
        ParallelRunner, StaticScheduler,
    };
}

/// Operation object for parallel thread execution with explicit broadcast/join control.
pub struct ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    pool: &'a mut ThreadPool,
    function: F,
    did_broadcast: bool,
    did_join: bool,
}

impl<'a, F> ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    /// Create a new ForThreadsOperation (internal use by ThreadPool)
    pub(crate) fn new(pool: &'a mut ThreadPool, function: F) -> Self {
        Self {
            pool,
            function,
            did_broadcast: false,
            did_join: false,
        }
    }

    /// Broadcast the work to all threads without waiting for completion.
    /// This is safe to call multiple times - subsequent calls are no-ops.
    pub fn broadcast(&mut self) {
        if self.did_broadcast {
            return; // No need to broadcast again
        }

        extern "C" fn trampoline<F>(ctx: *mut c_void, thread_index: usize, colocation_index: usize)
        where
            F: Fn(usize, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(thread_index, colocation_index);
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_unsafe_for_threads(self.pool.inner, trampoline::<F>, ctx);
            self.did_broadcast = true;
        }
    }

    /// Wait for all threads to complete their work.
    /// If broadcast() hasn't been called yet, this will call it first.
    pub fn join(&mut self) {
        if !self.did_broadcast {
            self.broadcast();
        }
        if self.did_join {
            return; // No need to join again
        }
        unsafe {
            fu_pool_unsafe_join(self.pool.inner);
            self.did_join = true;
        }
    }
}

impl<'a, F> Drop for ForThreadsOperation<'a, F>
where
    F: Fn(usize, usize) + Sync,
{
    fn drop(&mut self) {
        self.join();
    }
}

/// Operation object for parallel task execution with static load balancing.
pub struct ForNOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForNOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            task_index: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                colocation_index,
            });
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_n(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Operation object for parallel task execution with dynamic work-stealing.
pub struct ForNDynamicOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForNDynamicOperation<'a, F>
where
    F: Fn(Prong) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            task_index: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                colocation_index,
            });
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_n_dynamic(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Operation object for parallel slice execution.
pub struct ForSlicesOperation<'a, F>
where
    F: Fn(Prong, usize) + Sync,
{
    pool: &'a mut ThreadPool,
    n: usize,
    function: F,
}

impl<'a, F> Drop for ForSlicesOperation<'a, F>
where
    F: Fn(Prong, usize) + Sync,
{
    fn drop(&mut self) {
        extern "C" fn trampoline<F>(
            ctx: *mut c_void,
            first_index: usize,
            count: usize,
            thread_index: usize,
            colocation_index: usize,
        ) where
            F: Fn(Prong, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(
                Prong {
                    task_index: first_index,
                    thread_index,
                    colocation_index,
                },
                count,
            );
        }

        unsafe {
            let ctx = &self.function as *const F as *mut c_void;
            fu_pool_for_slices(self.pool.inner, self.n, trampoline::<F>, ctx);
        }
    }
}

/// Spawns a pool with the specified number of threads.
pub fn spawn(threads: usize) -> ThreadPool {
    ThreadPool::try_spawn(threads).expect("Failed to spawn ThreadPool")
}

/// Spawns a named pool with the specified number of threads.
pub fn named_spawn(name: &str, threads: usize) -> ThreadPool {
    ThreadPool::try_named_spawn(name, threads).expect("Failed to spawn named ThreadPool")
}

/// Standalone function to distribute `n` similar duration calls between threads.
pub fn for_n<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong) + Sync,
{
    let _operation = pool.for_n(n, function);
    // Operation executes and joins in its destructor
}

/// Standalone function to execute `n` uneven tasks on all threads.
pub fn for_n_dynamic<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong) + Sync,
{
    let _operation = pool.for_n_dynamic(n, function);
    // Operation executes and joins in its destructor
}

/// Standalone function to distribute `n` tasks in slices.
pub fn for_slices<F>(pool: &mut ThreadPool, n: usize, function: F)
where
    F: Fn(Prong, usize) + Sync,
{
    let _operation = pool.for_slices(n, function);
    // Operation executes and joins in its destructor
}

/// Helper function to visit every element exactly once with mutable access.
pub fn for_each_prong_mut<T, F>(pool: &mut ThreadPool, data: &mut [T], function: F)
where
    T: Send + Sync,
    F: Fn(&mut T, Prong) + Sync + Send,
{
    let ptr = SyncMutPtr::new(data.as_mut_ptr());
    let n = data.len();

    let _operation = pool.for_n(n, move |prong| {
        let item = unsafe { &mut *ptr.get(prong.task_index) };
        function(item, prong);
    });
}

/// Helper function to visit every element exactly once with dynamic work-stealing.
pub fn for_each_prong_mut_dynamic<T, F>(pool: &mut ThreadPool, data: &mut [T], function: F)
where
    T: Send + Sync,
    F: Fn(&mut T, Prong) + Sync + Send,
{
    let ptr = SyncMutPtr::new(data.as_mut_ptr());
    let n = data.len();

    let _operation = pool.for_n_dynamic(n, move |prong| {
        let item = unsafe { &mut *ptr.get(prong.task_index) };
        function(item, prong);
    });
}

/// Splits a range of tasks into fair-sized chunks for parallel distribution.
///
/// The first `(tasks % threads)` chunks have size `ceil(tasks / threads)`.
/// The remaining chunks have size `floor(tasks / threads)`.
///
/// This ensures optimal load balancing across threads with minimal size variance.
/// See: <https://lemire.me/blog/2025/05/22/dividing-an-array-into-fair-sized-chunks/>
#[derive(Debug, Clone)]
pub struct IndexedSplit {
    quotient: usize,
    remainder: usize,
}

impl IndexedSplit {
    /// Creates a new indexed split for distributing tasks across threads.
    ///
    /// # Arguments
    ///
    /// * `tasks_count` - Total number of tasks to distribute
    /// * `threads_count` - Number of threads to distribute across (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `threads_count` is zero.
    pub fn new(tasks_count: usize, threads_count: usize) -> Self {
        assert!(threads_count > 0, "Threads count must be greater than zero");
        Self {
            quotient: tasks_count / threads_count,
            remainder: tasks_count % threads_count,
        }
    }

    /// Returns the range for a specific thread index.
    pub fn get(&self, thread_index: usize) -> core::ops::Range<usize> {
        let begin = self.quotient * thread_index + thread_index.min(self.remainder);
        let count = self.quotient + if thread_index < self.remainder { 1 } else { 0 };
        begin..(begin + count)
    }
}

#[cfg(test)]
#[cfg(feature = "std")]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;
    use std::vec;
    use std::vec::Vec;

    #[inline]
    fn hw_threads() -> usize {
        count_logical_cores().max(1)
    }

    #[test]
    fn capabilities() {
        let caps = capabilities_string();
        std::println!("Capabilities: {caps:?}");
        assert!(caps.is_some());
    }

    #[test]
    fn system_info() {
        let cores = count_logical_cores();
        let numa = count_numa_nodes();
        let colocations = count_colocations();
        let qos = count_quality_levels();

        std::println!("Cores: {cores}, NUMA: {numa}, Colocations: {colocations}, QoS: {qos}");
        assert!(cores > 0);
    }

    #[test]
    fn spawn_and_basic_info() {
        let pool = spawn(2);
        assert_eq!(pool.threads(), 2);
        assert!(pool.colocations() > 0);
    }

    #[test]
    fn for_threads_dispatch() {
        let count_threads = hw_threads();
        let mut pool = spawn(count_threads);

        let visited: Arc<Vec<AtomicBool>> =
            Arc::new((0..count_threads).map(|_| AtomicBool::new(false)).collect());
        let visited_ref = Arc::clone(&visited);

        {
            let _op = pool.for_threads(move |thread_index, _colocation| {
                if thread_index < visited_ref.len() {
                    visited_ref[thread_index].store(true, Ordering::Relaxed);
                }
            });
        } // Operation executes in destructor

        for (i, flag) in visited.iter().enumerate() {
            assert!(
                flag.load(Ordering::Relaxed),
                "thread {i} never reached the callback"
            );
        }
    }

    #[test]
    fn for_n_static_scheduling() {
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(hw_threads());

        let visited: Arc<Vec<AtomicBool>> = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        for_n(&mut pool, EXPECTED_PARTS, move |prong| {
            let task_index = prong.task_index;
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "static scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn for_n_dynamic_scheduling() {
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(hw_threads());

        let visited: Arc<Vec<AtomicBool>> = Arc::new(
            (0..EXPECTED_PARTS)
                .map(|_| AtomicBool::new(false))
                .collect(),
        );
        let duplicate = Arc::new(AtomicBool::new(false));
        let visited_ref = Arc::clone(&visited);
        let duplicate_ref = Arc::clone(&duplicate);

        for_n_dynamic(&mut pool, EXPECTED_PARTS, move |prong| {
            let task_index = prong.task_index;
            if visited_ref[task_index].swap(true, Ordering::Relaxed) {
                duplicate_ref.store(true, Ordering::Relaxed);
            }
        });

        assert!(
            !duplicate.load(Ordering::Relaxed),
            "dynamic scheduling produced duplicate task IDs"
        );
        for flag in visited.iter() {
            assert!(flag.load(Ordering::Relaxed));
        }
    }

    #[test]
    fn for_each_mut() {
        const ELEMENTS: usize = 1000;
        let mut pool = spawn(hw_threads());
        let mut data = std::vec![0u64; ELEMENTS];

        for_each_prong_mut(&mut pool, &mut data, |x, prong| {
            *x = prong.task_index as u64 * 2;
        });

        for (i, &value) in data.iter().enumerate() {
            assert_eq!(value, i as u64 * 2);
        }
    }

    #[test]
    fn closure_objects() {
        let mut pool = spawn(hw_threads());
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        // Test that the operation object properly executes on drop
        {
            let _op = pool.for_n(1000, move |_prong| {
                counter_ref.fetch_add(1, Ordering::Relaxed);
            });
        } // Operation executes here in the destructor

        // Now the operation should have completed
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }

    #[test]
    fn explicit_broadcast_join() {
        let mut pool = spawn(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let mut operation = pool.for_threads(move |_thread_index, _colocation| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(10)); // Simulate work
        });

        // Broadcast work to threads but don't wait yet
        operation.broadcast();

        // Do some other work while threads are running
        thread::sleep(Duration::from_millis(5));

        // Now wait for completion
        operation.join();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn join_without_explicit_broadcast() {
        let mut pool = spawn(4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let mut operation = pool.for_threads(move |_thread_index, _colocation| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
        });

        // Join without calling broadcast first - should work
        operation.join();
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn pinned_allocator_creation() {
        let numa_count = count_numa_nodes();
        assert!(numa_count > 0, "System should have at least one NUMA node");

        // Test valid NUMA node
        let allocator = PinnedAllocator::new(0).expect("NUMA node 0 should be available");
        assert_eq!(allocator.numa_node(), 0);

        // Test invalid NUMA node
        let invalid_allocator = PinnedAllocator::new(numa_count + 10);
        assert!(
            invalid_allocator.is_none(),
            "Invalid NUMA node should return None"
        );
    }

    #[test]
    fn basic_allocation() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let allocation = allocator
            .allocate(1024)
            .expect("Failed to allocate 1024 bytes");

        assert_eq!(allocation.allocated_bytes(), 1024);
        assert_eq!(allocation.numa_node(), 0);

        // Test that we can write to the memory
        let slice = allocation.as_slice();
        assert_eq!(slice.len(), 1024);
    }

    #[test]
    fn allocate_zero_bytes() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let allocation = allocator.allocate(0);
        assert!(
            allocation.is_none(),
            "Allocating 0 bytes should return None"
        );
    }

    #[test]
    fn allocate_at_least() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let allocation = allocator
            .allocate_at_least(1000)
            .expect("Failed to allocate at least 1000 bytes");

        assert!(allocation.allocated_bytes() >= 1000);
        assert_eq!(allocation.numa_node(), 0);

        // bytes_per_page should be set to something reasonable
        if allocation.bytes_per_page() > 0 {
            assert!(allocation.bytes_per_page() >= 512); // Reasonable minimum page size
        }
    }

    #[test]
    fn pinned_vec_creation() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let vec = PinnedVec::<i32>::new_in(allocator);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert_eq!(vec.numa_node(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn pinned_vec_with_capacity() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let vec = PinnedVec::<i32>::with_capacity_in(allocator, 10).expect("Failed to create vec");
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);
        assert_eq!(vec.numa_node(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn pinned_vec_push_pop() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);

        // Test push
        vec.push(42).expect("Failed to push");
        assert_eq!(vec.len(), 1);
        assert!(!vec.is_empty());
        assert_eq!(vec[0], 42);

        vec.push(100).expect("Failed to push");
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[1], 100);

        // Test pop
        assert_eq!(vec.pop(), Some(100));
        assert_eq!(vec.len(), 1);
        assert_eq!(vec.pop(), Some(42));
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.pop(), None);
    }

    #[test]
    fn pinned_vec_indexing() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(10).expect("Failed to push");
        vec.push(20).expect("Failed to push");
        vec.push(30).expect("Failed to push");

        // Test indexing
        assert_eq!(vec[0], 10);
        assert_eq!(vec[1], 20);
        assert_eq!(vec[2], 30);

        // Test mutable indexing
        vec[1] = 25;
        assert_eq!(vec[1], 25);
    }

    #[test]
    fn pinned_vec_clear() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(1).expect("Failed to push");
        vec.push(2).expect("Failed to push");
        vec.push(3).expect("Failed to push");

        assert_eq!(vec.len(), 3);
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[test]
    fn pinned_vec_insert_remove() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(1).expect("Failed to push");
        vec.push(3).expect("Failed to push");

        // Insert in the middle
        vec.insert(1, 2).expect("Failed to insert");
        assert_eq!(vec.len(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);

        // Remove from the middle
        let removed = vec.remove(1);
        assert_eq!(removed, 2);
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 3);
    }

    #[test]
    fn pinned_vec_reserve() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        assert_eq!(vec.capacity(), 0);

        vec.reserve(10).expect("Failed to reserve");
        assert!(vec.capacity() >= 10);
        assert_eq!(vec.len(), 0);

        // Adding elements shouldn't require new allocation
        for i in 0..10 {
            vec.push(i).expect("Failed to push");
        }
        assert_eq!(vec.len(), 10);
    }

    #[test]
    fn pinned_vec_extend_from_slice() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        let data = [1, 2, 3, 4, 5];

        vec.extend_from_slice(&data).expect("Failed to extend");
        assert_eq!(vec.len(), 5);
        for (i, &value) in data.iter().enumerate() {
            assert_eq!(vec[i], value);
        }
    }

    #[test]
    fn parallel_slice_static_for_each() {
        let mut pool = spawn(hw_threads());
        let data: Vec<usize> = (0..512).collect();
        let total = AtomicUsize::new(0);

        (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each(|value| {
                total.fetch_add(*value, Ordering::Relaxed);
            });

        assert_eq!(total.load(Ordering::Relaxed), data.iter().sum());
    }

    #[test]
    fn parallel_slice_mut_dynamic_for_each() {
        let mut pool = spawn(hw_threads());
        let mut data: Vec<usize> = (0..256).collect();

        (&mut data[..])
            .into_par_iter()
            .with_schedule(&mut pool, DynamicScheduler)
            .for_each(|value| {
                *value *= 2;
            });

        for (i, v) in data.iter().enumerate() {
            assert_eq!(*v, i * 2);
        }
    }

    #[test]
    fn parallel_slice_zip_sum() {
        let mut pool = spawn(hw_threads());
        let a: Vec<usize> = (0..128).collect();
        let b: Vec<usize> = (0..128).rev().collect();
        let sums: Arc<Vec<AtomicUsize>> =
            Arc::new((0..hw_threads()).map(|_| AtomicUsize::new(0)).collect());
        let shared = Arc::clone(&sums);

        (&a[..])
            .into_par_iter()
            .zip((&b[..]).into_par_iter())
            .with_pool(&mut pool)
            .for_each_with_prong(|(lhs, rhs), prong| {
                shared[prong.thread_index % shared.len()].fetch_add(lhs + rhs, Ordering::Relaxed);
            });

        let total: usize = sums.iter().map(|v| v.load(Ordering::Relaxed)).sum();
        let expected: usize = a.iter().zip(b.iter()).map(|(x, y)| x + y).sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn parallel_exact_iter_dispatch() {
        let mut pool = spawn(hw_threads());
        let mut values = vec![0usize; 256];
        let ptr = SyncMutPtr::new(values.as_mut_ptr());
        (0..values.len())
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each_with_prong(|index, prong| {
                let slot = unsafe { &mut *ptr.get(prong.task_index) };
                *slot = index * index;
            });

        for (idx, val) in values.iter().enumerate() {
            assert_eq!(*val, idx * idx);
        }
    }

    #[test]
    fn round_robin_parallel_mut() {
        let mut pool = spawn(hw_threads());
        let mut rr_vec =
            RoundRobinVec::<usize>::with_capacity_per_colocation(8).expect("round robin vec");

        // Populate evenly
        for value in 0..32 {
            rr_vec.push(value).expect("push");
        }

        rr_vec
            .par_iter_mut()
            .with_pool(&mut pool)
            .for_each(|value| {
                *value += 1;
            });

        for index in 0..rr_vec.len() {
            assert_eq!(rr_vec.get(index), Some(&(index + 1)));
        }
    }

    #[test]
    fn scratch_reduction_collects_sum() {
        let mut pool = spawn(hw_threads());
        let data: Vec<usize> = (0..1024).collect();
        let mut scratch = vec![0usize; pool.threads()];

        (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .fold_with_scratch(scratch.as_mut_slice(), |slot, value, _| {
                *slot += *value;
            });

        let total: usize = scratch.iter().sum();
        let expected: usize = data.iter().sum();
        assert_eq!(total, expected);
    }

    #[test]
    fn pinned_vec_iterators() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        for i in 0..5 {
            vec.push(i).expect("Failed to push");
        }

        // Test immutable iterator
        let collected: Vec<i32> = vec.iter().copied().collect();
        let expected = Vec::from([0, 1, 2, 3, 4]);
        assert_eq!(collected, expected);

        // Test mutable iterator
        for value in vec.iter_mut() {
            *value *= 2;
        }
        assert_eq!(vec[0], 0);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 4);
        assert_eq!(vec[3], 6);
        assert_eq!(vec[4], 8);
    }

    #[test]
    fn pinned_vec_slices() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        for i in 0..5 {
            vec.push(i).expect("Failed to push");
        }

        // Test as_slice
        let slice = vec.as_slice();
        assert_eq!(slice.len(), 5);
        assert_eq!(slice[2], 2);

        // Test as_mut_slice
        let mut_slice = vec.as_mut_slice();
        mut_slice[2] = 99;
        assert_eq!(vec[2], 99);
    }

    #[test]
    fn pinned_vec_growth() {
        let allocator = PinnedAllocator::new(0).expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);

        // Push many elements to test growth
        for i in 0..100 {
            vec.push(i).expect("Failed to push");
        }

        assert_eq!(vec.len(), 100);
        for i in 0..100 {
            assert_eq!(vec[i], i as i32);
        }
    }

    #[test]
    fn pinned_vec_invalid_numa_node() {
        let numa_count = count_numa_nodes();
        let allocator = PinnedAllocator::new(numa_count + 1);
        assert!(allocator.is_none());
    }

    #[test]
    fn sync_const_ptr() {
        let data = Vec::from([1, 2, 3, 4, 5]);
        let sync_ptr = SyncConstPtr::new(data.as_ptr());

        unsafe {
            assert_eq!(*sync_ptr.get(0), 1);
            assert_eq!(*sync_ptr.get(2), 3);
            assert_eq!(*sync_ptr.get(4), 5);
        }

        assert_eq!(sync_ptr.as_ptr(), data.as_ptr());
    }

    #[test]
    fn sync_const_ptr_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SyncConstPtr<i32>>();
        assert_sync::<SyncConstPtr<i32>>();
    }

    #[test]
    fn pinned_vec_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<PinnedVec<i32>>();
        assert_sync::<PinnedVec<i32>>();
    }

    #[test]
    fn indexed_split() {
        // Test basic split
        let split = IndexedSplit::new(10, 3);
        assert_eq!(split.get(0), 0..4); // `ceil(10/3)` = 4
        assert_eq!(split.get(1), 4..7); // `floor(10/3)` = 3
        assert_eq!(split.get(2), 7..10); // `floor(10/3)` = 3

        // Test even split
        let split = IndexedSplit::new(12, 3);
        assert_eq!(split.get(0), 0..4);
        assert_eq!(split.get(1), 4..8);
        assert_eq!(split.get(2), 8..12);

        // Test edge cases
        let split = IndexedSplit::new(0, 2);
        assert_eq!(split.get(0), 0..0);
        assert_eq!(split.get(1), 0..0);

        let split = IndexedSplit::new(1, 2);
        assert_eq!(split.get(0), 0..1);
        assert_eq!(split.get(1), 1..1);
    }

    #[test]
    #[should_panic(expected = "Threads count must be greater than zero")]
    fn indexed_split_zero_threads() {
        IndexedSplit::new(10, 0);
    }
}
