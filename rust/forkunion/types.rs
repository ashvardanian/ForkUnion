//! Portable building blocks - cache-line padding, spin mutexes, prongs, sync pointers, and splits.
//!
//! Pure logic with no FFI; mirrors the C++ `types` header.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, Ordering};

/// Default alignment for preventing false sharing between threads.
///
/// Set to 128 bytes to account for adjacent cache-line prefetching on modern CPUs.
/// This matches the C++ `default_alignment_k` constant defined in `forkunion.hpp`.
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
/// use forkunion::{CacheAligned, ThreadPool, Topology};
///
/// let topology = Topology::new().unwrap();
/// let mut pool = ThreadPool::try_spawn(&topology, 4).unwrap();
/// let data: Vec<usize> = (0..1000).collect();
///
/// // Each thread gets its own cache-aligned accumulator
/// let mut scratch: Vec<CacheAligned<usize>> =
///     (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();
///
/// // No false sharing during parallel reduction
/// for value in &data {
///     let tid = *value % pool.threads_count();
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
/// use forkunion::*;
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
    /// use forkunion::*;
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
    /// use forkunion::*;
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
    /// use forkunion::*;
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
    /// use forkunion::*;
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
    /// use forkunion::*;
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
    /// use forkunion::*;
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
/// use forkunion::*;
///
/// let mutex = SpinMutex::new(42);
/// let mut guard = mutex.lock();
/// *guard = 100;
/// ```
pub type SpinMutex<T> = BasicSpinMutex<T, true>;

/// A "prong" - the tip of a "fork" - pinning a "task" to a "thread" within a "compute domain".
///
/// A `Prong` represents a single unit of work that connects:
/// - A **task** (what work to do) - identified by `task_index`
/// - A **thread** (which CPU thread is executing it) - identified by `thread_index`
/// - A **compute domain** (the same-QoS core cluster it runs on) - identified by `compute_domain_index`
///
/// This metadata is essential for topology-aware algorithms, debugging parallel execution,
/// and understanding load distribution across the thread pool.
#[derive(Copy, Clone, Debug)]
pub struct Prong {
    /// The logical index of the task being processed (0-based)
    pub task_index: usize,
    /// The physical thread executing this task (0-based)
    pub thread_index: usize,
    /// The compute domain this thread belongs to (a same-QoS core cluster within a memory domain)
    pub compute_domain_index: usize,
}

/// A thread-safe wrapper for raw pointers used in parallel operations.
///
/// # Safety
/// This wrapper is only safe when used with NUMA-aware thread pools where
/// each thread accesses different memory locations - different memory domains.

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
/// use forkunion::*;
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
mod tests {
    use super::*;

    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use std::vec::Vec;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn guard_lifecycle_exclusive() {
        let topology = Topology::new().unwrap();
        // On exclusive pools the work is dispatched at guard construction:
        // the caller can overlap its own work and poll `is_complete`.
        let mut pool =
            ThreadPool::try_spawn_with_exclusivity(&topology, 4, CallerExclusivity::Exclusive)
                .expect("Failed to create exclusive thread pool");
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let broadcast_function = move |_thread_index: usize, _compute_domain: usize| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
        };
        let mut operation = pool.for_threads(&broadcast_function);
        assert!(
            operation.generation().is_some(),
            "Exclusive pools dispatch at construction"
        );
        assert_eq!(
            operation.generation().unwrap() & 1,
            1,
            "Generation tokens are always odd"
        );

        // Exclusive pools complete without the caller contributing a slice, so `join`
        // blocks purely on the workers; afterwards the completion query must observe them
        // done - a deterministic check with no busy-wait and no timing assumptions.
        operation.join();
        assert!(
            operation.is_complete(),
            "join must leave the operation complete"
        );
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn guard_lifecycle_inclusive() {
        let topology = Topology::new().unwrap();
        // On inclusive pools the dispatch is deferred to `join`, where the
        // calling thread contributes its own slice.
        let mut pool = spawn(&topology, 4);
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_ref = Arc::clone(&counter);

        let broadcast_function = move |_thread_index: usize, _compute_domain: usize| {
            counter_ref.fetch_add(1, Ordering::Relaxed);
        };
        let mut operation = pool.for_threads(&broadcast_function);
        assert!(
            operation.generation().is_none(),
            "Inclusive pools defer dispatch to join"
        );
        assert!(!operation.is_complete());
        assert_eq!(
            counter.load(Ordering::Relaxed),
            0,
            "No work must start before join"
        );

        operation.join();
        assert!(operation.is_complete());
        assert_eq!(counter.load(Ordering::Relaxed), 4);
    }

    #[cfg_attr(miri, ignore)]
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
