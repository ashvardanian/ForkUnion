//! Portable building blocks - cache-line padding, spin mutexes, task ranges, sync pointers, and splits.
//!
//! Pure logic with no FFI; mirrors the C++ `types` header.

use core::cell::UnsafeCell;
use core::ffi::c_int;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicBool, Ordering};

/// Why a call into the C core failed, in the vocabulary the core itself uses.
///
/// Mirrors `fu_status_t` value-for-value, so the FFI boundary only retypes. `Unrecognized` covers a
/// status a newer core reports and this build has no name for.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// No reason was reported, or one this build does not name.
    Unknown = -1,
    /// An allocation or mapping failed; a smaller request may succeed.
    BadAlloc = -2,
    /// A fixed ceiling was reached, so a smaller request will not help either.
    CapacityExhausted = -3,
    /// An argument was malformed, out of range, or would overflow a byte count.
    InvalidArgument = -4,
    /// The handles or the pool kind cannot serve this call together.
    ConfigMismatch = -5,
    /// The pool is already spawned; terminate it first.
    AlreadySpawned = -6,
    /// The pool was never spawned.
    NotSpawned = -7,
    /// The OS declined to create a thread - a resource limit, or permissions.
    ThreadRefused = -8,
    /// The machine could not be described; transient if its CPU set changed mid-probe.
    TopologyUnavailable = -9,
    /// A privileged operation was declined.
    PermissionDenied = -10,
    /// This build or this machine has no such facility.
    Unsupported = -11,
    /// A status this binding has no name for.
    Unrecognized = i32::MIN,
}

impl Status {
    /// Maps a raw `fu_status_t`, keeping an unnamed one rather than guessing.
    fn from_raw(raw: i32) -> Self {
        match raw {
            -1 => Self::Unknown,
            -2 => Self::BadAlloc,
            -3 => Self::CapacityExhausted,
            -4 => Self::InvalidArgument,
            -5 => Self::ConfigMismatch,
            -6 => Self::AlreadySpawned,
            -7 => Self::NotSpawned,
            -8 => Self::ThreadRefused,
            -9 => Self::TopologyUnavailable,
            -10 => Self::PermissionDenied,
            -11 => Self::Unsupported,
            _ => Self::Unrecognized,
        }
    }

    /// Static, English description; mirrors `fu_status_to_string`.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Unknown => "the core reported no reason",
            Self::BadAlloc => "an allocation failed",
            Self::CapacityExhausted => "a fixed capacity was exhausted",
            Self::InvalidArgument => "an argument was rejected",
            Self::ConfigMismatch => "the handles cannot serve this call together",
            Self::AlreadySpawned => "the pool is already spawned",
            Self::NotSpawned => "the pool was never spawned",
            Self::ThreadRefused => "the OS declined to create a thread",
            Self::TopologyUnavailable => "the machine could not be described",
            Self::PermissionDenied => "a privileged operation was declined",
            Self::Unsupported => "this build has no such facility",
            Self::Unrecognized => "an unrecognized status",
        }
    }
}

/// A failure, whether the core reported it or the binding caught it before the call.
///
/// `detail` names the symbol or the argument, which is what turns a status into a diagnosis.
/// Both fields are plain data, so nothing allocates on the failure path - which is what lets this
/// crate stay `no_std`.
#[derive(Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Error {
    /// The identity of the failure, shared with the C interface.
    pub status: Status,
    /// The detail the status alone cannot carry.
    pub detail: &'static str,
}

impl Error {
    /// A failure the binding caught before the call crossed, under the status the core would use.
    pub(crate) const fn new(status: Status, detail: &'static str) -> Self {
        Self { status, detail }
    }

    /// Turns a raw status into a `Result`, naming `detail` on failure.
    pub(crate) fn check(raw: c_int, detail: &'static str) -> Result<()> {
        if raw == 0 {
            return Ok(());
        }
        Err(Error::new(Status::from_raw(raw), detail))
    }
}

impl core::fmt::Debug for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "Error({:?}, {:?})", self.status, self.detail)
    }
}

/// The result of every fallible call in this crate.
pub type Result<T> = core::result::Result<T, Error>;

impl core::fmt::Display for Error {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "{}: {}", self.detail, self.status.describe())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Bytes occupied by `count` elements of `element_bytes` each, refusing a product that would wrap.
pub fn bytes_for_elements(count: usize, element_bytes: usize) -> Result<usize> {
    match count.checked_mul(element_bytes) {
        Some(bytes) => Ok(bytes),
        None => Err(Error::new(
            Status::InvalidArgument,
            "the element count times the element size would wrap",
        )),
    }
}

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
/// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
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
#[repr(align(128))]
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
    #[must_use]
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
    #[must_use]
    pub fn lock(&self) -> BasicSpinMutexGuard<'_, T, PAUSE> {
        loop {
            // The only store in the loop, so contenders spin on a shared line rather than
            // taking it exclusive on every attempt.
            if !self.locked.swap(true, Ordering::Acquire) {
                return BasicSpinMutexGuard { mutex: self };
            }
            while self.locked.load(Ordering::Relaxed) {
                if PAUSE {
                    core::hint::spin_loop();
                }
            }
        }
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
    /// if let Some(mut guard) = mutex.lock_if_free() {
    ///     *guard = 42;
    ///     println!("Lock acquired and value set");
    /// } else {
    ///     println!("Lock is currently held by another thread");
    /// };
    /// ```
    pub fn lock_if_free(&self) -> Option<BasicSpinMutexGuard<'_, T, PAUSE>> {
        if !self.locked.swap(true, Ordering::Acquire) {
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn get(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }

    /// Returns a mutable reference to the protected data.
    ///
    /// This method is rarely needed since the guard implements `DerefMut`.
    #[must_use]
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

/// A half-open `[first, first + count)` run of task indices - the "what work" of a slice dispatch.
///
/// Iterable, so a callback reads `for task in range` rather than rebuilding the bounds. An idle
/// thread receives a range with `count == 0`, which every dispatch still calls exactly once.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TasksRange {
    /// The first task index in the run.
    pub first: usize,
    /// How many tasks the run covers; zero means an idle thread.
    pub count: usize,
}

impl TasksRange {
    /// One past the last task index, so `first..end()` is the half-open span.
    #[must_use]
    pub fn end(&self) -> usize {
        self.first + self.count
    }

    /// Whether the run covers no tasks at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The run as a `Range`, for indexing a slice or driving a `for` loop.
    #[must_use]
    pub fn range(&self) -> core::ops::Range<usize> {
        self.first..self.end()
    }
}

impl IntoIterator for TasksRange {
    type Item = usize;
    type IntoIter = core::ops::Range<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.range()
    }
}

/// One thread, situated in one compute domain - the "where I am" every callback receives.
///
/// Every dispatch hands its callback two things: the work, and this. A callback placing memory
/// reads `compute_domain` to find the node it runs on; one indexing per-thread scratch reads
/// `thread`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ThreadInDomain {
    /// The physical thread executing the work (0-based).
    pub thread: usize,
    /// The compute domain the thread is pinned to - a same-QoS core cluster within a memory domain.
    pub compute_domain: usize,
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
    /// Wraps a raw pointer; every obligation is discharged at [`get`](Self::get).
    #[must_use]
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
    #[inline]
    #[must_use]
    pub unsafe fn get(&self, index: usize) -> &T {
        &*self.ptr.add(index)
    }

    /// Returns the raw pointer.
    #[inline]
    #[must_use]
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
    #[must_use]
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
    #[inline]
    #[must_use]
    pub unsafe fn get(&self, index: usize) -> *mut T {
        self.ptr.add(index)
    }

    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *mut T {
        self.ptr
    }
}

unsafe impl<T> Send for SyncMutPtr<T> {}
unsafe impl<T> Sync for SyncMutPtr<T> {}

/// Splits a range of tasks into fair-sized runs for parallel distribution.
///
/// The first `(tasks % threads)` runs have size `ceil(tasks / threads)`.
/// The remaining runs have size `floor(tasks / threads)`.
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
    #[must_use]
    pub fn new(tasks_count: usize, threads_count: usize) -> Self {
        assert!(threads_count > 0, "Threads count must be greater than zero");
        Self {
            quotient: tasks_count / threads_count,
            remainder: tasks_count % threads_count,
        }
    }

    /// Returns the range for a specific thread index.
    #[inline]
    #[must_use]
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
            ThreadPool::spawn_with_exclusivity(&topology, 4, CallerExclusivity::Exclusive)
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

        // Join blocks on the workers; the completion query is only meaningful after it.
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
        let _ = IndexedSplit::new(10, 0);
    }
}
