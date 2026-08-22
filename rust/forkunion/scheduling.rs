//! The thread pool and its dispatch primitives - spawn, scoped joins, parallel loops - and the
//! measured memory fabric those pools can harvest.
//!
//! Owns the `fu_pool_*` and `fu_fabric_*` FFI; mirrors the C++ `flat`/`distributed` scheduling layer.

use crate::parallel::{ParallelIterator, ParallelSchedule};
use crate::topology::{
    CallerExclusivity, Capabilities, ComputeDomain, Error, MemoryDomain, Topology,
};
use crate::types::{IndexedSplit, Prong, SyncMutPtr};
use core::ffi::{c_char, c_int, c_void};

extern "C" {
    fn fu_pool_new(name: *const c_char, allowed: u32) -> *mut c_void;
    fn fu_pool_delete(pool: *mut c_void);
    fn fu_pool_spawn(
        topology: *mut c_void,
        pool: *mut c_void,
        threads: usize,
        exclusivity: c_int,
    ) -> c_int;
    fn fu_pool_spawn_on(
        topology: *mut c_void,
        pool: *mut c_void,
        compute_domain_index: usize,
        threads: usize,
        exclusivity: c_int,
    ) -> c_int;
    fn fu_pool_caller_exclusivity(pool: *mut c_void) -> c_int;
    fn fu_pool_compute_domains_count(pool: *mut c_void) -> usize;
    fn fu_pool_threads_count_in(pool: *mut c_void, compute_domain_index: usize) -> usize;
    fn fu_pool_threads_count(pool: *mut c_void) -> usize;
    fn fu_pool_locate_thread_in(
        pool: *mut c_void,
        global_thread_index: usize,
        compute_domain_index: usize,
    ) -> usize;
    fn fu_pool_sleep(pool: *mut c_void, micros: usize);
    fn fu_pool_terminate(pool: *mut c_void);

    #[allow(dead_code)]
    fn fu_pool_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    );
    fn fu_pool_for_slices(
        pool: *mut c_void,
        n: usize,
        callback: extern "C" fn(*mut c_void, usize, usize, usize, usize),
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

    fn fu_pool_unsafe_for_threads(
        pool: *mut c_void,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    ) -> usize;
    fn fu_pool_is_complete(pool: *mut c_void, generation: usize) -> c_int;
    fn fu_pool_unsafe_join(pool: *mut c_void, generation: usize);
    fn fu_pool_capabilities(pool: *mut c_void) -> u32;

    fn fu_fabric_new() -> *mut c_void;
    fn fu_fabric_delete(fabric: *mut c_void);
    fn fu_fabric_harvest(topology: *mut c_void, pool: *mut c_void, fabric: *mut c_void) -> c_int;
    fn fu_fabric_memory_latency(
        fabric: *mut c_void,
        compute_domain_index: usize,
        memory_domain_index: usize,
    ) -> usize;
    fn fu_fabric_memory_bandwidth(
        fabric: *mut c_void,
        compute_domain_index: usize,
        memory_domain_index: usize,
    ) -> usize;
    fn fu_fabric_memory_distance(
        fabric: *mut c_void,
        compute_domain_index: usize,
        memory_domain_index: usize,
    ) -> usize;
    fn fu_fabric_memory_level_in(fabric: *mut c_void, memory_domain_index: usize) -> usize;
    fn fu_fabric_memory_levels_count(fabric: *mut c_void) -> usize;
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
/// use forkunion::*;
///
/// // Create a thread pool with 4 threads
/// let topology = Topology::new().unwrap();
/// let mut pool = spawn(&topology, 4);
///
/// // Execute work on each thread
/// pool.for_threads(&|thread_index, compute_domain_index| {
///     println!("Thread {} on compute_domain {}", thread_index, compute_domain_index);
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
///
/// # Generation Tokens
///
/// Every dispatch is identified by an always-odd `usize` generation token. The safe
/// `for_threads` API wraps it inside a [`BroadcastJoin`] guard: on `Exclusive` pools the
/// work starts at construction and can be polled with `is_complete`; on `Inclusive` pools
/// it runs at `join`/`Drop`, where the calling thread contributes its own slice. Raw
/// token-level access is available through the `unsafe_for_threads`/`unsafe_join` pair.
pub struct ThreadPool {
    inner: *mut c_void,
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}

impl ThreadPool {
    pub fn try_spawn_with_exclusivity(
        topology: &Topology,
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(topology, None, threads, exclusivity)
    }

    pub fn try_named_spawn_with_exclusivity(
        topology: &Topology,
        name: Option<&str>,
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        Self::try_named_spawn_with_capabilities(
            topology,
            name,
            threads,
            exclusivity,
            Capabilities::ALL,
        )
    }

    /// As [`try_named_spawn_with_exclusivity`](Self::try_named_spawn_with_exclusivity), but constrains
    /// the pool to `allowed`: clear a waiter bit to force a lower-priority busy-wait, or clear
    /// [`Capabilities::PLACE_MEMORY_ON_DOMAIN`] to force the flat (non-NUMA) pool.
    pub fn try_named_spawn_with_capabilities(
        topology: &Topology,
        name: Option<&str>,
        threads: usize,
        exclusivity: CallerExclusivity,
        allowed: Capabilities,
    ) -> Result<Self, Error> {
        if threads == 0 {
            return Err(Error::InvalidParameter);
        }

        // The buffer must outlive the `fu_pool_new` call, so it is declared
        // before taking the pointer that crosses the FFI boundary.
        let mut name_buffer = [0u8; 16];
        let name_ptr = if let Some(name_str) = name {
            let name_bytes = name_str.as_bytes();
            let copy_len = core::cmp::min(name_bytes.len(), 15); // Leave space for null terminator
            name_buffer[..copy_len].copy_from_slice(&name_bytes[..copy_len]);
            // name_buffer[copy_len] is already 0 from initialization
            name_buffer.as_ptr() as *const c_char
        } else {
            core::ptr::null()
        };

        unsafe {
            let inner = fu_pool_new(name_ptr, allowed.0);
            if inner.is_null() {
                return Err(Error::CreationFailed);
            }

            let success = fu_pool_spawn(topology.raw(), inner, threads, exclusivity as c_int);
            if success == 0 {
                fu_pool_delete(inner);
                return Err(Error::SpawnFailed);
            }

            Ok(Self { inner })
        }
    }

    /// Spawns a pool pinned to a single compute domain (a same-QoS core cluster).
    ///
    /// The pool's threads and NUMA-local allocations stay on `compute_domain_index`, in
    /// `0..compute_domains_count()`. Spawn one such pool per compute domain and coordinate them
    /// from a single thread with the generation-token API (`for_threads` guards or the
    /// raw `unsafe_for_threads`/`is_complete`/`unsafe_join`). On builds without NUMA,
    /// only compute domain 0 is valid.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    /// // One pool per compute domain, sized to that domain's core count.
    /// let topology = Topology::new().unwrap();
    /// let pools: Vec<ThreadPool> = (0..topology.compute_domains_count())
    ///     .map(|c| ThreadPool::try_spawn_on(&topology, c, topology.logical_cores_count_in(ComputeDomain(c)).max(1), CallerExclusivity::Exclusive).unwrap())
    ///     .collect();
    /// assert_eq!(pools.len(), topology.compute_domains_count());
    /// ```
    pub fn try_spawn_on(
        topology: &Topology,
        compute_domain_index: usize,
        threads: usize,
        exclusivity: CallerExclusivity,
    ) -> Result<Self, Error> {
        Self::try_spawn_on_with_capabilities(
            topology,
            compute_domain_index,
            threads,
            exclusivity,
            Capabilities::ALL,
        )
    }

    /// As [`try_spawn_on`](Self::try_spawn_on), but constrains the colocated pool to `allowed`.
    pub fn try_spawn_on_with_capabilities(
        topology: &Topology,
        compute_domain_index: usize,
        threads: usize,
        exclusivity: CallerExclusivity,
        allowed: Capabilities,
    ) -> Result<Self, Error> {
        if threads == 0 {
            return Err(Error::InvalidParameter);
        }
        unsafe {
            let inner = fu_pool_new(core::ptr::null(), allowed.0);
            if inner.is_null() {
                return Err(Error::CreationFailed);
            }
            let success = fu_pool_spawn_on(
                topology.raw(),
                inner,
                compute_domain_index,
                threads,
                exclusivity as c_int,
            );
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
    /// use forkunion::*;
    ///
    /// // Create a pool that uses 4 threads total (3 spawned + caller)
    /// let topology = Topology::new().unwrap();
    /// let pool = ThreadPool::try_spawn(&topology, 4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads_count(), 4);
    /// ```
    pub fn try_spawn(topology: &Topology, threads: usize) -> Result<Self, Error> {
        Self::try_spawn_with_exclusivity(topology, threads, CallerExclusivity::Inclusive)
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let pool = ThreadPool::try_named_spawn(&topology, "worker_pool", 4).expect("Failed to create thread pool");
    /// assert_eq!(pool.threads_count(), 4);
    /// ```
    pub fn try_named_spawn(topology: &Topology, name: &str, threads: usize) -> Result<Self, Error> {
        Self::try_named_spawn_with_exclusivity(
            topology,
            Some(name),
            threads,
            CallerExclusivity::Inclusive,
        )
    }

    /// Returns whether the calling thread participates in the workload.
    ///
    /// Queries the pool directly rather than caching, so it stays correct across
    /// `terminate` and re-spawning with a different exclusivity.
    pub fn caller_exclusivity(&self) -> CallerExclusivity {
        match unsafe { fu_pool_caller_exclusivity(self.inner) } {
            0 => CallerExclusivity::Inclusive,
            _ => CallerExclusivity::Exclusive,
        }
    }

    /// Returns the capabilities this pool was spawned with.
    ///
    /// Queries the pool directly rather than caching, so it reflects the allow-mask the
    /// pool actually honors - the intersection of the requested mask and what the machine offers.
    pub fn capabilities(&self) -> Capabilities {
        Capabilities(unsafe { fu_pool_capabilities(self.inner) })
    }

    /// Returns the number of thread compute_domains in the pool.
    ///
    /// Compute domains group threads sharing a memory domain, QoS level, and cache hierarchy.
    /// This information is useful for NUMA-aware load balancing and memory allocation.
    pub fn compute_domains_count(&self) -> usize {
        unsafe { fu_pool_compute_domains_count(self.inner) }
    }

    /// Returns the number of threads in a specific compute_domain.
    ///
    /// This method is useful for NUMA-aware load balancing, allowing you to understand
    /// how many threads are available in each compute_domain group.
    ///
    /// # Arguments
    ///
    /// * `compute_domain_index` - The compute_domain to query (0-based)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let pool = spawn(&topology, 8);
    /// let total_compute_domains = pool.compute_domains_count();
    ///
    /// for compute_domain_index in 0..total_compute_domains {
    ///     let thread_count = pool.threads_count_in(compute_domain_index);
    ///     println!("ComputeDomain {} has {} threads", compute_domain_index, thread_count);
    /// }
    /// ```
    pub fn threads_count_in(&self, compute_domain_index: usize) -> usize {
        unsafe { fu_pool_threads_count_in(self.inner, compute_domain_index) }
    }

    /// Returns the number of threads in the pool.
    pub fn threads_count(&self) -> usize {
        unsafe { fu_pool_threads_count(self.inner) }
    }

    /// Converts a global thread index to a local thread index within a compute_domain.
    ///
    /// This is useful for distributed thread pools where threads are grouped into
    /// compute domains (same-QoS core clusters). The local index can be used for
    /// per-compute_domain data structures or algorithms.
    ///
    /// # Arguments
    ///
    /// * `global_thread_index` - The global thread index to convert
    /// * `compute_domain_index` - The compute_domain to get the local index for
    ///
    /// # Returns
    ///
    /// The local thread index within the specified compute_domain.
    pub fn locate_thread_in(
        &self,
        global_thread_index: usize,
        compute_domain_index: usize,
    ) -> usize {
        unsafe { fu_pool_locate_thread_in(self.inner, global_thread_index, compute_domain_index) }
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
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

    /// Executes a function on each thread of the pool, returning a [`BroadcastJoin`] guard.
    ///
    /// The guard's lifecycle is keyed on the pool's exclusivity:
    /// - `CallerExclusivity::Exclusive`: the work is dispatched immediately at construction;
    ///   the caller can overlap its own work, poll `is_complete`, and `join` (or drop) waits.
    /// - `CallerExclusivity::Inclusive`: the dispatch is deferred to `join` (or drop), where
    ///   the calling thread contributes its own slice - a deferred blocking call.
    ///
    /// # Arguments
    ///
    /// * `function` - Closure reference executed on each thread, receiving
    ///   `(thread_index, compute_domain_index)`; borrowed for the guard's lifetime.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
    /// pool.for_threads(&|thread_index, compute_domain_index| {
    ///     println!("Thread {} on compute_domain {}", thread_index, compute_domain_index);
    /// })
    /// .join();
    /// ```
    pub fn for_threads<'fork, F>(&mut self, function: &'fork F) -> BroadcastJoin<'_, 'fork, F>
    where
        F: Fn(usize, usize) + Sync,
    {
        BroadcastJoin::new(self, function)
    }

    /// Runs `function` on every thread and blocks until all of them finish.
    ///
    /// The ergonomic common case: unlike [`for_threads`](Self::for_threads), this takes
    /// the closure **by value** and joins internally, so there is no `&` binding or guard
    /// to manage. Reach for `for_threads` only when you want to overlap the caller's own
    /// work with the pool and poll [`BroadcastJoin::is_complete`] before joining.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
    /// let counter = std::sync::atomic::AtomicUsize::new(0);
    /// pool.broadcast(|_thread_index, _compute_domain_index| {
    ///     counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    /// });
    /// assert_eq!(counter.load(std::sync::atomic::Ordering::Relaxed), pool.threads_count());
    /// ```
    pub fn broadcast<F>(&mut self, function: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        // `function` lives on this frame for the whole dispatch-and-join, so the borrow
        // handed to the pool cannot dangle - and `join` runs before it drops.
        BroadcastJoin::new(self, &function).join();
    }

    /// Runs `body` with a [`Scope`] that can broadcast work borrowing local data and answer
    /// read-only topology queries, joining every dispatch before returning.
    ///
    /// The scope holds the pool by shared reference, so a worker closure can query it
    /// (`threads_count_in`, `locate_thread_in`) *and* borrow the same stack values the caller owns
    /// - the borrow conflict that otherwise forces a [`crate::SafePtr`] smuggle. Because each
    /// [`Scope::broadcast`] blocks until it joins, those borrows can never outlive the work.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
    /// let counter = SpinMutex::new(0usize);
    /// pool.scope(|scope| {
    ///     scope.broadcast(|thread_index, compute_domain_index| {
    ///         let _local = scope.locate_thread_in(thread_index, compute_domain_index);
    ///         *counter.lock() += 1;
    ///     });
    /// });
    /// assert_eq!(*counter.lock(), pool.threads_count());
    /// ```
    pub fn scope<F, R>(&mut self, body: F) -> R
    where
        F: FnOnce(&Scope) -> R,
    {
        let scope = Scope { pool: self };
        body(&scope)
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
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

    /// Splits `data` into one contiguous chunk per thread and runs `function` on each in
    /// parallel, blocking until all threads finish.
    ///
    /// Each thread receives an **exclusive** `&mut` sub-slice, so no interior mutability,
    /// `Mutex`, or raw pointers are needed at the call site: the chunks partition `data`
    /// and therefore never alias, and the synchronous join keeps every borrow inside
    /// `data`'s lifetime. This is the safe replacement for hand-rolled [`crate::SafePtr`] scatter.
    ///
    /// Built on [`for_threads`](Self::for_threads), so `function` runs once per thread even when
    /// its chunk is empty - unlike [`for_slices`](Self::for_slices), which skips an empty range.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
    /// let mut data = vec![0u64; 1000];
    /// pool.for_slices_mut(&mut data, |_thread_index, chunk| {
    ///     for value in chunk {
    ///         *value += 1;
    ///     }
    /// });
    /// assert!(data.iter().all(|&value| value == 1));
    /// ```
    pub fn for_slices_mut<T, F>(&mut self, data: &mut [T], function: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync,
    {
        let threads = self.threads_count();
        let split = IndexedSplit::new(data.len(), threads);
        let base = SyncMutPtr::new(data.as_mut_ptr()); // ? `Sync` wrapper for the disjoint scatter
        let function = &function;
        let scatter = move |thread_index: usize, _compute_domain_index: usize| {
            let range = split.get(thread_index);
            // SAFETY: disjoint in-bounds ranges per thread, joined before `data`'s borrow ends.
            // `get` only offsets - an empty trailing range lands one past the end, never deref'd.
            let chunk =
                unsafe { core::slice::from_raw_parts_mut(base.get(range.start), range.len()) };
            function(thread_index, chunk);
        };
        BroadcastJoin::new(self, &scatter).join();
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = spawn(&topology, 4);
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

    /// Dispatches `callback` on every thread without blocking, returning the generation token.
    ///
    /// This is the raw C-ABI mirror for building custom orchestration; prefer the safe
    /// [`BroadcastJoin`] guard returned by `for_threads`.
    ///
    /// # Safety
    ///
    /// - Only one thread may operate the pool at a time, and only one dispatch may be in
    ///   flight: `unsafe_join` must complete before the next dispatch or pool destruction.
    /// - `callback` and `context` must remain valid until `unsafe_join` returns.
    pub unsafe fn unsafe_for_threads(
        &self,
        callback: extern "C" fn(*mut c_void, usize, usize),
        context: *mut c_void,
    ) -> usize {
        fu_pool_unsafe_for_threads(self.inner, callback, context)
    }

    /// Returns true if the given generation has completed on all threads.
    ///
    /// A `true` result also guarantees visibility of every contributor's writes. On
    /// `Inclusive` pools this can only turn `true` once `unsafe_join` contributes the
    /// calling thread's slice, so the poll-then-join pattern is reserved for
    /// `Exclusive` pools.
    pub fn is_complete(&self, generation: usize) -> bool {
        unsafe { fu_pool_is_complete(self.inner, generation) != 0 }
    }

    /// Blocks until the given generation completes; idempotent for joined generations.
    ///
    /// On `Inclusive` pools this also executes the calling thread's slice of the work.
    ///
    /// # Safety
    ///
    /// Must be called on the thread operating the pool, with the dispatched callback
    /// and context still valid.
    pub unsafe fn unsafe_join(&self, generation: usize) {
        fu_pool_unsafe_join(self.inner, generation)
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

/// The measured memory fabric - what this process observed, as opposed to the structure a
/// [`Topology`] declares. Two query families: edge queries `(initiator, target)` describe one
/// interconnect link; medium queries `(target)` describe the memory pool itself, independent of
/// any initiator.
///
/// Completes the `try_harvest` pipeline: a [`Topology`] is harvested first and stays immutable
/// (and shareable), a [`ThreadPool`] spawns on it, and the fabric then harvests through that
/// pool's pinned workers, snapshotting what it needs so the topology may be dropped after.
/// Before a harvest every query answers 0, and
/// [`memory_levels_count`](Self::memory_levels_count) answers 1.
///
/// # Examples
///
/// ```rust,no_run
/// use forkunion::*;
/// let topology = Topology::new().unwrap();
/// let mut pool = spawn(&topology, 4);
/// let mut fabric = Fabric::new().unwrap();
/// if fabric.try_harvest(&topology, &mut pool) {
///     let local = topology.local_memory_of(ComputeDomain(0));
///     assert!(fabric.memory_latency(ComputeDomain(0), local) > 0);
/// }
/// ```
pub struct Fabric {
    inner: *mut c_void,
}

unsafe impl Send for Fabric {}
unsafe impl Sync for Fabric {}

impl Fabric {
    /// Creates an empty, unharvested fabric.
    pub fn new() -> Result<Fabric, Error> {
        let inner = unsafe { fu_fabric_new() };
        if inner.is_null() {
            return Err(Error::CreationFailed);
        }
        Ok(Fabric { inner })
    }

    /// Measures the memory fabric through the pool's pinned workers, replacing any previous
    /// harvest; the `topology` is only read.
    ///
    /// Returns `false` on allocation failure, or for a pool whose workers are not pinned per
    /// domain - one restricted below [`Capabilities::PLACE_MEMORY_ON_DOMAIN`] or pinned via
    /// [`ThreadPool::try_spawn_on`]; the fabric is then left empty, never half-written. Not
    /// thread-safe: it dispatches on the pool and rebuilds `self`, so call it between task
    /// batches. Expect seconds of runtime on large fabrics.
    pub fn try_harvest(&mut self, topology: &Topology, pool: &mut ThreadPool) -> bool {
        unsafe { fu_fabric_harvest(topology.raw(), pool.inner, self.inner) != 0 }
    }

    /// Returns the measured dependent-load latency (nanoseconds) on an edge - the best recording;
    /// 0 before a harvest, for an edge no worker could reach, or an out-of-range index.
    pub fn memory_latency(
        &self,
        compute_domain: ComputeDomain,
        memory_domain: MemoryDomain,
    ) -> usize {
        unsafe { fu_fabric_memory_latency(self.inner, compute_domain.get(), memory_domain.get()) }
    }

    /// Returns the measured saturated read bandwidth (MB/s) on an edge, streamed by all the
    /// initiator domain's workers at once - the best recording; 0 if unreached or out of range.
    pub fn memory_bandwidth(
        &self,
        compute_domain: ComputeDomain,
        memory_domain: MemoryDomain,
    ) -> usize {
        unsafe { fu_fabric_memory_bandwidth(self.inner, compute_domain.get(), memory_domain.get()) }
    }

    /// Returns the relative access distance on an edge (10 = local, per the SLIT convention):
    /// the measured latency ratio to the initiator's local domain, clamped so local carries the
    /// row's minimum; unwalked edges fall back to 10-local / 20-remote.
    pub fn memory_distance(
        &self,
        compute_domain: ComputeDomain,
        memory_domain: MemoryDomain,
    ) -> usize {
        unsafe { fu_fabric_memory_distance(self.inner, compute_domain.get(), memory_domain.get()) }
    }

    /// Returns the derived speed class of a memory domain (lower = faster: HBM < DDR < CXL),
    /// keyed by the best bandwidth any initiator sustains to it, ties split by the best latency.
    pub fn memory_level_in(&self, memory_domain: MemoryDomain) -> usize {
        unsafe { fu_fabric_memory_level_in(self.inner, memory_domain.get()) }
    }

    /// Returns the number of distinct derived memory tiers, the memory-axis twin of
    /// [`Topology::compute_levels_count`]; 1 on single-tier systems and before a harvest.
    pub fn memory_levels_count(&self) -> usize {
        unsafe { fu_fabric_memory_levels_count(self.inner) }
    }
}

impl Drop for Fabric {
    fn drop(&mut self) {
        unsafe { fu_fabric_delete(self.inner) };
    }
}

/// A synchronization guard that waits for all threads to finish the broadcasted closure.
///
/// The lifecycle is keyed on the pool's exclusivity:
/// - On `CallerExclusivity::Exclusive` pools the closure is dispatched at **construction**:
///   the workers start immediately, the caller can overlap its own work, poll
///   `is_complete`, and `join` (or `Drop`) waits for completion.
/// - On `CallerExclusivity::Inclusive` pools the dispatch is deferred to **join** (or
///   `Drop`), where the calling thread contributes its own slice of the work.
///
/// The closure is borrowed rather than owned, so its address stays stable while worker
/// threads hold a pointer to it, and the guard itself remains freely movable.
pub struct BroadcastJoin<'pool, 'fork, F>
where
    F: Fn(usize, usize) + Sync,
{
    pool: &'pool mut ThreadPool,
    function: &'fork F,
    state: BroadcastState,
}

/// Where a broadcast stands in its dispatch-then-join lifecycle; tokens are always odd.
enum BroadcastState {
    Pending,
    Dispatched(usize),
    Joined(usize),
}

impl BroadcastState {
    fn generation(&self) -> Option<usize> {
        match *self {
            Self::Pending => None,
            Self::Dispatched(generation) | Self::Joined(generation) => Some(generation),
        }
    }
}

impl<'pool, 'fork, F> BroadcastJoin<'pool, 'fork, F>
where
    F: Fn(usize, usize) + Sync,
{
    /// Create a new BroadcastJoin (internal use by ThreadPool)
    pub(crate) fn new(pool: &'pool mut ThreadPool, function: &'fork F) -> Self {
        let mut operation = Self {
            pool,
            function,
            state: BroadcastState::Pending,
        };
        if operation.pool.caller_exclusivity() == CallerExclusivity::Exclusive {
            operation.dispatch();
        }
        operation
    }

    /// Dispatches if it has not already, and yields the generation token either way.
    fn dispatch(&mut self) -> usize {
        if let Some(generation) = self.state.generation() {
            return generation;
        }

        extern "C" fn trampoline<F>(
            context: *mut c_void,
            thread_index: usize,
            compute_domain_index: usize,
        ) where
            F: Fn(usize, usize) + Sync,
        {
            let function = unsafe { &*(context as *const F) };
            function(thread_index, compute_domain_index);
        }

        let generation = unsafe {
            let context = self.function as *const F as *mut c_void;
            fu_pool_unsafe_for_threads(self.pool.inner, trampoline::<F>, context)
        };
        self.state = BroadcastState::Dispatched(generation);
        generation
    }

    /// The generation token of this broadcast; always odd once dispatched.
    pub fn generation(&self) -> Option<usize> {
        self.state.generation()
    }

    /// True once the dispatched generation has fully completed on all threads.
    ///
    /// A `true` result also guarantees visibility of every contributor's writes. On
    /// `Inclusive` pools this can only turn `true` once `join` contributes the calling
    /// thread's slice, so the poll-then-join pattern is reserved for `Exclusive` pools.
    pub fn is_complete(&self) -> bool {
        match self.state {
            BroadcastState::Pending => false,
            BroadcastState::Dispatched(generation) => unsafe {
                fu_pool_is_complete(self.pool.inner, generation) != 0
            },
            BroadcastState::Joined(_) => true,
        }
    }

    /// Wait for all threads to complete their work.
    /// On `Inclusive` pools this dispatches the work first and contributes the caller's slice.
    /// Idempotent - subsequent calls are no-ops.
    pub fn join(&mut self) {
        let generation = self.dispatch();
        if matches!(self.state, BroadcastState::Joined(_)) {
            return;
        }
        unsafe { fu_pool_unsafe_join(self.pool.inner, generation) };
        self.state = BroadcastState::Joined(generation);
    }
}

impl<F> Drop for BroadcastJoin<'_, '_, F>
where
    F: Fn(usize, usize) + Sync,
{
    fn drop(&mut self) {
        self.join();
    }
}

/// A borrow-scoped handle to a thread pool, yielded by [`ThreadPool::scope`].
///
/// Holding the pool by shared reference is what lets a worker closure both query the pool
/// (`threads_count_in`, `locate_thread_in`) and borrow the caller's stack data at the same time -
/// the borrow conflict that otherwise forces a [`crate::SafePtr`] smuggle. Every [`Scope::broadcast`]
/// joins before returning, so those borrows are always valid.
pub struct Scope<'pool> {
    pool: &'pool ThreadPool,
}

impl Scope<'_> {
    /// Total number of worker threads in the pool.
    pub fn threads_count(&self) -> usize {
        self.pool.threads_count()
    }

    /// Number of compute domains the pool spans.
    pub fn compute_domains_count(&self) -> usize {
        self.pool.compute_domains_count()
    }

    /// Number of threads pinned to the given compute domain.
    pub fn threads_count_in(&self, compute_domain_index: usize) -> usize {
        self.pool.threads_count_in(compute_domain_index)
    }

    /// Local index of a global thread within its compute domain.
    pub fn locate_thread_in(
        &self,
        global_thread_index: usize,
        compute_domain_index: usize,
    ) -> usize {
        self.pool
            .locate_thread_in(global_thread_index, compute_domain_index)
    }

    /// Broadcasts `function` to every thread and blocks until all of them finish.
    ///
    /// The closure is borrowed for the dispatch and joined before this returns, so it may freely
    /// borrow the stack data enclosing the [`ThreadPool::scope`] call.
    pub fn broadcast<F>(&self, function: F)
    where
        F: Fn(usize, usize) + Sync,
    {
        extern "C" fn trampoline<F>(
            context: *mut c_void,
            thread_index: usize,
            compute_domain_index: usize,
        ) where
            F: Fn(usize, usize) + Sync,
        {
            let function = unsafe { &*(context as *const F) };
            function(thread_index, compute_domain_index);
        }

        // SAFETY: `function` outlives the dispatch because we join before returning, and the
        // enclosing `scope` holds the pool by `&mut`, so no other dispatch overlaps this one.
        unsafe {
            let context = &function as *const F as *mut c_void;
            let generation = self.pool.unsafe_for_threads(trampoline::<F>, context);
            self.pool.unsafe_join(generation);
        }
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
            compute_domain_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                compute_domain_index,
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
            compute_domain_index: usize,
        ) where
            F: Fn(Prong) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(Prong {
                task_index,
                thread_index,
                compute_domain_index,
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
            compute_domain_index: usize,
        ) where
            F: Fn(Prong, usize) + Sync,
        {
            let f = unsafe { &*(ctx as *const F) };
            f(
                Prong {
                    task_index: first_index,
                    thread_index,
                    compute_domain_index,
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
        scratch_len >= pool.threads_count(),
        "scratch space must cover all threads"
    );
    let scratch_ptr = SyncMutPtr::new(scratch.as_mut_ptr());
    iterator.drive(pool, schedule, &move |item, prong| {
        debug_assert!(prong.thread_index < scratch_len);
        let slot = unsafe { &mut *scratch_ptr.get(prong.thread_index) };
        fold(slot, item, prong);
    });
}

/// Spawns a pool with the specified number of threads.
pub fn spawn(topology: &Topology, threads: usize) -> ThreadPool {
    ThreadPool::try_spawn(topology, threads).expect("Failed to spawn ThreadPool")
}

/// Spawns a named pool with the specified number of threads.
pub fn named_spawn(topology: &Topology, name: &str, threads: usize) -> ThreadPool {
    ThreadPool::try_named_spawn(topology, name, threads).expect("Failed to spawn named ThreadPool")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::tests::hw_threads;
    use crate::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    use std::vec::Vec;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn spawn_and_basic_info() {
        let topology = Topology::new().unwrap();
        let pool = spawn(&topology, 2);
        assert_eq!(pool.threads_count(), 2);
        assert!(pool.compute_domains_count() > 0);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn caller_exclusivity_query() {
        let topology = Topology::new().unwrap();
        // The pool is the single source of truth, queried live (not cached).
        let inclusive =
            ThreadPool::try_spawn_with_exclusivity(&topology, 2, CallerExclusivity::Inclusive)
                .unwrap();
        assert_eq!(inclusive.caller_exclusivity(), CallerExclusivity::Inclusive);

        let exclusive =
            ThreadPool::try_spawn_with_exclusivity(&topology, 2, CallerExclusivity::Exclusive)
                .unwrap();
        assert_eq!(exclusive.caller_exclusivity(), CallerExclusivity::Exclusive);

        // The default `spawn` is inclusive
        assert_eq!(
            spawn(&topology, 2).caller_exclusivity(),
            CallerExclusivity::Inclusive
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn per_compute_domain_pools() {
        let topology = Topology::new().unwrap();
        // One pool per compute_domain, each pinned to its node; drive them from this thread.
        let compute_domains = topology.compute_domains_count();
        assert!(compute_domains >= 1);

        let mut pools: Vec<ThreadPool> = (0..compute_domains)
            .map(|c| {
                let cores = topology.logical_cores_count_in(ComputeDomain(c)).max(1);
                ThreadPool::try_spawn_on(&topology, c, cores, CallerExclusivity::Exclusive)
                    .expect("failed to spawn per-compute_domain pool")
            })
            .collect();

        // Each pinned pool independently runs work across its own threads.
        for pool in &mut pools {
            let counter = AtomicUsize::new(0);
            pool.broadcast(|_thread_index, _compute_domain_index| {
                counter.fetch_add(1, Ordering::Relaxed);
            });
            assert_eq!(counter.load(Ordering::Relaxed), pool.threads_count());
        }

        // Out-of-range compute_domain must fail cleanly, not panic.
        assert!(ThreadPool::try_spawn_on(
            &topology,
            compute_domains + 100,
            2,
            CallerExclusivity::Exclusive
        )
        .is_err());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_threads_dispatch() {
        let topology = Topology::new().unwrap();
        let count_threads = hw_threads();
        let mut pool = spawn(&topology, count_threads);

        let visited: Arc<Vec<AtomicBool>> =
            Arc::new((0..count_threads).map(|_| AtomicBool::new(false)).collect());
        let visited_ref = Arc::clone(&visited);

        {
            let broadcast_function = move |thread_index: usize, _compute_domain: usize| {
                if thread_index < visited_ref.len() {
                    visited_ref[thread_index].store(true, Ordering::Relaxed);
                }
            };
            let _operation = pool.for_threads(&broadcast_function);
        } // Operation executes in destructor

        for (i, flag) in visited.iter().enumerate() {
            assert!(
                flag.load(Ordering::Relaxed),
                "thread {i} never reached the callback"
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn broadcast_owns_and_blocks() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let counter = AtomicUsize::new(0);
        // Owned closure, no `&` and no guard - borrows `counter` from this frame safely.
        pool.broadcast(|_thread_index, _compute_domain_index| {
            counter.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(counter.load(Ordering::Relaxed), pool.threads_count());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_slices_mut_partitions_disjointly() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let total = 10_000usize;
        let mut data: Vec<usize> = (0..total).collect();

        // Each thread squares its own exclusive chunk - no SafePtr, no Mutex.
        pool.for_slices_mut(&mut data, |_thread_index, chunk| {
            for value in chunk {
                *value *= *value;
            }
        });

        for (index, &value) in data.iter().enumerate() {
            assert_eq!(
                value,
                index * index,
                "element {index} not processed exactly once"
            );
        }
    }

    /// `for_slices_mut`'s chunk arithmetic without a pool, so Miri reaches it.
    #[test]
    fn slice_chunks_stay_in_bounds_when_threads_outnumber_elements() {
        for total in [0usize, 1, 2, 3] {
            let mut data: Vec<u8> = (0..total as u8).collect();
            let threads = 8usize;
            let split = IndexedSplit::new(data.len(), threads);
            let base = SyncMutPtr::new(data.as_mut_ptr());

            let mut visited = 0usize;
            for thread_index in 0..threads {
                let range = split.get(thread_index);
                // SAFETY: mirrors `for_slices_mut` - trailing empty ranges only offset.
                let chunk =
                    unsafe { core::slice::from_raw_parts_mut(base.get(range.start), range.len()) };
                for value in chunk.iter_mut() {
                    *value = value.wrapping_add(1);
                }
                visited += range.len();
            }
            assert_eq!(visited, total, "chunks must cover {total} elements");
            assert!(
                data.iter()
                    .enumerate()
                    .all(|(index, &value)| value == index as u8 + 1),
                "every element visited once for {total} elements"
            );
        }
    }

    /// Once per thread even when a thread draws no elements.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_slices_mut_calls_every_thread_on_short_slices() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let threads = pool.threads_count();

        for total in [0usize, 1, 2] {
            let calls = Arc::new(AtomicUsize::new(0));
            let covered = Arc::new(AtomicUsize::new(0));
            let mut data: Vec<usize> = (0..total).collect();

            {
                let calls = Arc::clone(&calls);
                let covered = Arc::clone(&covered);
                pool.for_slices_mut(&mut data, move |_thread_index, chunk| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    covered.fetch_add(chunk.len(), Ordering::Relaxed);
                    for value in chunk {
                        *value += 100;
                    }
                });
            }

            assert_eq!(
                calls.load(Ordering::Relaxed),
                threads,
                "one call per thread"
            );
            assert_eq!(
                covered.load(Ordering::Relaxed),
                total,
                "chunks cover the slice"
            );
            for (index, &value) in data.iter().enumerate() {
                assert_eq!(value, index + 100, "element {index} not processed once");
            }
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_n_static_scheduling() {
        let topology = Topology::new().unwrap();
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(&topology, hw_threads());

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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_n_dynamic_scheduling() {
        let topology = Topology::new().unwrap();
        const EXPECTED_PARTS: usize = 1_000;
        let mut pool = spawn(&topology, hw_threads());

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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn for_each_mut() {
        let topology = Topology::new().unwrap();
        const ELEMENTS: usize = 1000;
        let mut pool = spawn(&topology, hw_threads());
        let mut data = std::vec![0u64; ELEMENTS];

        for_each_prong_mut(&mut pool, &mut data, |x, prong| {
            *x = prong.task_index as u64 * 2;
        });

        for (i, &value) in data.iter().enumerate() {
            assert_eq!(value, i as u64 * 2);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn closure_objects() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn generation_multi_pool_completion() {
        let topology = Topology::new().unwrap();
        // Two independent exclusive pools run side by side; join both, then query each.
        let count_threads = hw_threads();
        let mut pool_a = ThreadPool::try_spawn_with_exclusivity(
            &topology,
            count_threads,
            CallerExclusivity::Exclusive,
        )
        .expect("Failed to create pool_a");
        let mut pool_b = ThreadPool::try_spawn_with_exclusivity(
            &topology,
            count_threads,
            CallerExclusivity::Exclusive,
        )
        .expect("Failed to create pool_b");

        let visited_a: Vec<AtomicBool> =
            (0..count_threads).map(|_| AtomicBool::new(false)).collect();
        let visited_b: Vec<AtomicBool> =
            (0..count_threads).map(|_| AtomicBool::new(false)).collect();

        let work_a = |thread_index: usize, _compute_domain: usize| {
            visited_a[thread_index].store(true, Ordering::Relaxed);
        };
        let work_b = |thread_index: usize, _compute_domain: usize| {
            visited_b[thread_index].store(true, Ordering::Relaxed);
        };

        let mut operation_a = pool_a.for_threads(&work_a);
        let mut operation_b = pool_b.for_threads(&work_b);

        operation_a.join();
        operation_b.join();
        assert!(
            operation_a.is_complete(),
            "pool_a generation must be complete after join"
        );
        assert!(
            operation_b.is_complete(),
            "pool_b generation must be complete after join"
        );

        for i in 0..count_threads {
            assert!(
                visited_a[i].load(Ordering::Relaxed),
                "Thread {i} in pool_a not visited"
            );
            assert!(
                visited_b[i].load(Ordering::Relaxed),
                "Thread {i} in pool_b not visited"
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn fabric_harvest_fills_edges() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let mut fabric = Fabric::new().unwrap();

        // An unharvested fabric answers zeros and a single tier.
        assert_eq!(fabric.memory_latency(ComputeDomain(0), MemoryDomain(0)), 0);
        assert_eq!(fabric.memory_levels_count(), 1);

        if !fabric.try_harvest(&topology, &mut pool) {
            return; // ? A flat pool without domain placement has no fabric to walk
        }
        // Every reachable edge must carry sane observations; emulated-NUMA guests may
        // measure equal local and remote costs, so nothing stronger is asserted.
        let local = topology.local_memory_of(ComputeDomain(0));
        assert!(fabric.memory_latency(ComputeDomain(0), local) > 0);
        assert!(fabric.memory_bandwidth(ComputeDomain(0), local) > 0);
        assert_eq!(fabric.memory_distance(ComputeDomain(0), local), 10);
        assert!(fabric.memory_levels_count() >= 1);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn generation_raw_unsafe_api() {
        let topology = Topology::new().unwrap();
        // The raw C-ABI mirror: an `unsafe_for_threads` dispatch returning an odd token,
        // polled with the safe `is_complete`, and joined with `unsafe_join`.
        let count_threads = hw_threads();
        let pool = ThreadPool::try_spawn_with_exclusivity(
            &topology,
            count_threads,
            CallerExclusivity::Exclusive,
        )
        .expect("Failed to create exclusive thread pool");

        let counter = AtomicUsize::new(0);

        extern "C" fn trampoline(
            context: *mut c_void,
            _thread_index: usize,
            _compute_domain_index: usize,
        ) {
            let counter = unsafe { &*(context as *const AtomicUsize) };
            counter.fetch_add(1, Ordering::Relaxed);
        }

        let generation = unsafe {
            pool.unsafe_for_threads(trampoline, &counter as *const AtomicUsize as *mut c_void)
        };
        assert_eq!(generation & 1, 1, "Generation tokens are always odd");

        // Join blocks on the workers; query completion only after.
        unsafe { pool.unsafe_join(generation) };
        assert!(
            pool.is_complete(generation),
            "join must leave the generation complete"
        );
        assert_eq!(counter.load(Ordering::Relaxed), count_threads);
    }
}
