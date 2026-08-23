//! Rayon-style parallel iterators built on top of the thread pool.
//!
//! Pure logic layered over `scheduling`; no FFI of its own.

use crate::allocators::{DomainAllocator, PinnedVec};
use crate::scheduling::{fold_with_scratch, ThreadPool};
use crate::topology::{MemoryDomain, Topology};
use crate::types::{CacheAligned, Result, SyncMutPtr, TasksRange, ThreadInDomain};
use core::cell::UnsafeCell;
use core::marker::PhantomData;

/// Sync wrapper for single-write cells used in early-exit operations.
///
/// # Safety
///
/// This is safe because:
/// - Only one thread writes (enforced by AtomicBool in caller)
/// - Write happens-before any subsequent read (synchronized by atomic operations)
/// - Final read happens after all threads finish (enforced by drive() completion)
struct SyncOnceCell<T> {
    inner: UnsafeCell<Option<T>>,
}

unsafe impl<T> Sync for SyncOnceCell<T> {}

impl<T> SyncOnceCell<T> {
    const fn new() -> Self {
        Self {
            inner: UnsafeCell::new(None),
        }
    }

    /// SAFETY: Caller must ensure only one thread calls this
    unsafe fn set(&self, value: T) {
        *self.inner.get() = Some(value);
    }

    fn into_inner(self) -> Option<T> {
        self.inner.into_inner()
    }
}

/// Scheduler that uses static chunk assignment.
#[derive(Clone, Copy, Debug)]
pub struct StaticScheduler;

/// Scheduler that enables dynamic work stealing.
#[derive(Clone, Copy, Debug)]
pub struct DynamicScheduler;

pub trait ParallelSchedule: Copy {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(usize, ThreadInDomain) + Sync;

    /// Runs `function` over contiguous runs. The count is a real run length under static
    /// scheduling and always 1 under dynamic, which has no batched entry point.
    fn dispatch_slices<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(TasksRange, ThreadInDomain) + Sync;
}

impl ParallelSchedule for StaticScheduler {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(usize, ThreadInDomain) + Sync,
    {
        if tasks == 0 {
            return Ok(());
        }

        pool.for_n(tasks, function)
    }

    fn dispatch_slices<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(TasksRange, ThreadInDomain) + Sync,
    {
        if tasks == 0 {
            return Ok(());
        }

        pool.for_slices(tasks, function)
    }
}

impl ParallelSchedule for DynamicScheduler {
    fn dispatch<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(usize, ThreadInDomain) + Sync,
    {
        if tasks == 0 {
            return Ok(());
        }

        pool.for_n_dynamic(tasks, function)
    }

    /// Work-stealing hands out one task at a time, so every run is length 1 - the C core exposes
    /// no batched dynamic entry point. Use [`StaticScheduler`] when the batching is what you want.
    fn dispatch_slices<F>(&self, pool: &mut ThreadPool, tasks: usize, function: F) -> Result<()>
    where
        F: Fn(TasksRange, ThreadInDomain) + Sync,
    {
        self.dispatch(pool, tasks, move |task, at| {
            function(
                TasksRange {
                    first: task,
                    count: 1,
                },
                at,
            )
        })
    }
}

/// One thread's best candidate for an indexed search: the winning task index and its item.
type BestCandidate<T> = CacheAligned<Option<(usize, T)>>;

pub trait ParallelIterator: Sized {
    type Item;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync;

    fn drive_static<F>(self, pool: &mut ThreadPool, consumer: F) -> Result<()>
    where
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        self.drive(pool, StaticScheduler, &consumer)
    }

    fn drive_dynamic<F>(self, pool: &mut ThreadPool, consumer: F) -> Result<()>
    where
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        self.drive(pool, DynamicScheduler, &consumer)
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
    #[must_use]
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

    pub fn for_each<F>(self, function: F) -> Result<()>
    where
        F: Fn(I::Item, usize, ThreadInDomain) + Sync,
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;
        iterator.drive(pool, schedule, &move |item, task, at| {
            function(item, task, at)
        })
    }

    pub fn fold_with_scratch<T, F>(self, scratch: &mut [T], fold: F) -> Result<()>
    where
        T: Send,
        F: Fn(&mut T, I::Item, usize, ThreadInDomain) + Sync,
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;
        fold_with_scratch(pool, iterator, schedule, scratch, fold)
    }

    /// Parallel reduction with caller-provided scratch buffer.
    ///
    /// Reduces items in parallel by folding into per-thread accumulators,
    /// then combining results on the caller thread. Uses cache-aligned scratch
    /// to prevent false sharing. Indexes by thread_index (works with dynamic scheduling).
    ///
    /// # Arguments
    /// * `scratch` - Per-thread accumulators (must be `>= pool.threads_count()`)
    /// * `fold` - Function to accumulate items: `fn(&mut T, I::Item, usize, ThreadInDomain)`
    /// * `combine` - Function to merge two accumulators: `fn(T, T) -> T`
    ///
    /// # Returns
    /// The final reduced value of type `T`
    ///
    /// # Example
    /// ```
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..1000).collect();
    /// let mut scratch: Vec<CacheAligned<u64>> =
    ///     (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();
    ///
    /// let total = (&data[..]).into_par_iter().with_pool(&mut pool)
    ///     .reduce_with_scratch(
    ///         scratch.as_mut_slice(),
    ///         |acc, value, _, _| acc.0 += *value,
    ///         |a, b| a.0 += b.0,
    ///     );
    /// ```
    pub fn reduce_with_scratch<T, F, C>(self, scratch: &mut [T], fold: F, combine: C) -> Result<T>
    where
        T: Send + Default,
        F: Fn(&mut T, I::Item, usize, ThreadInDomain) + Sync,
        C: Fn(&mut T, T),
    {
        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;

        // Fold phase: accumulate into per-thread slots
        fold_with_scratch(pool, iterator, schedule, scratch, fold)?;

        // Combine phase: merge all slots into first slot in-place
        let (first, rest) = scratch
            .split_first_mut()
            .expect("scratch must not be empty");
        for slot in rest {
            let value = core::mem::take(slot);
            combine(first, value);
        }
        Ok(core::mem::take(first))
    }

    /// Fold with early-exit on error, using caller-provided scratch buffer.
    ///
    /// Similar to `fold_with_scratch`, but allows the fold function to return `Result`.
    /// Stops processing on the first error. Scratch buffers are indexed by `thread_index`.
    ///
    /// # Arguments
    ///
    /// * `scratch` - Per-thread accumulators (must be `>= pool.threads_count()`)
    /// * `fold` - Fallible fold function: `fn(&mut T, I::Item, usize, ThreadInDomain) -> core::result::Result<(), E>`
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all items were folded successfully
    /// - `Err(E)` with the first error encountered
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// fn checked_add(acc: &mut u64, value: &u64) -> core::result::Result<(), &'static str> {
    ///     *acc = acc.checked_add(*value).ok_or("overflow")?;
    ///     Ok(())
    /// }
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (1..100).collect();
    /// let mut scratch: Vec<CacheAligned<u64>> =
    ///     (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();
    ///
    /// let result = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .fold_with_scratch_fallible(scratch.as_mut_slice(), |acc, value, _, _| {
    ///         checked_add(&mut acc.0, value)
    ///     });
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn fold_with_scratch_fallible<T, F, E>(
        self,
        scratch: &mut [T],
        fold: F,
    ) -> Result<core::result::Result<(), E>>
    where
        T: Send,
        F: Fn(&mut T, I::Item, usize, ThreadInDomain) -> core::result::Result<(), E> + Sync,
        E: Send,
    {
        use core::sync::atomic::{AtomicBool, Ordering};

        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;

        let stop = AtomicBool::new(false);
        let first_err = SyncOnceCell::new();
        let scratch_ptr = SyncMutPtr::new(scratch.as_mut_ptr());

        iterator.drive(pool, schedule, &|item, task, at| {
            if stop.load(Ordering::Acquire) {
                return;
            }

            let slot = unsafe { &mut *scratch_ptr.get(at.thread) };

            if let Err(error) = fold(slot, item, task, at) {
                let already_stopped = stop.swap(true, Ordering::Release);
                if !already_stopped {
                    // SAFETY: Only one thread sets stop to true, so only one write
                    unsafe { first_err.set(error) };
                }
            }
        })?;

        // SAFETY: All worker threads finished, exclusive access
        Ok(match first_err.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }

    /// Executes a fallible operation on each item, stopping at the first error.
    ///
    /// Uses cooperative cancellation: once an error occurs, no further items are processed.
    /// Items already "in flight" may still complete, but new items won't start processing.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if all items were processed successfully or were skipped after stop
    /// - `Err(E)` with the first error encountered
    ///
    /// # Performance
    ///
    /// Overhead is one atomic load per item (~2% in compute-bound workloads).
    /// The atomic swap on error is negligible as it happens at most once.
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// fn validate(x: &u64) -> core::result::Result<(), &'static str> {
    ///     if *x < 100 { Ok(()) } else { Err("value too large") }
    /// }
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..50).collect();
    ///
    /// let result = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .for_each_fallible(|x, _, _| validate(x));
    ///
    /// assert!(result.is_ok());
    /// ```
    pub fn for_each_fallible<F, E>(self, function: F) -> Result<core::result::Result<(), E>>
    where
        F: Fn(I::Item, usize, ThreadInDomain) -> core::result::Result<(), E> + Sync,
        E: Send,
    {
        use core::sync::atomic::{AtomicBool, Ordering};

        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;

        let stop = AtomicBool::new(false);
        let first_err = SyncOnceCell::new();
        iterator.drive(pool, schedule, &|item, task, at| {
            if stop.load(Ordering::Acquire) {
                return;
            }

            if let Err(error) = function(item, task, at) {
                let already_stopped = stop.swap(true, Ordering::Release);
                if !already_stopped {
                    // SAFETY: Only one thread sets stop to true, so only one write
                    unsafe { first_err.set(error) };
                }
            }
        })?;

        // SAFETY: All worker threads finished, exclusive access
        Ok(match first_err.into_inner() {
            Some(error) => Err(error),
            None => Ok(()),
        })
    }

    /// Searches for the first element that matches a predicate (deterministic, by index).
    ///
    /// Returns the element with the smallest `task_index` among all matches.
    ///
    /// # Returns
    ///
    /// - `Some(item)` with the lowest index if any match was found
    /// - `None` if no item matched or the iterator was empty
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = vec![10, 20, 30, 20, 10];
    ///
    /// let found = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .find_first(|&&x| x == 20)
    ///     .unwrap();
    ///
    /// assert_eq!(found, Some(&20)); // Index 1, not 3
    /// ```
    pub fn find_first<P>(self, predicate: P) -> Result<Option<I::Item>>
    where
        I::Item: Send,
        P: Fn(&I::Item) -> bool + Sync,
    {
        // The index has to travel with the item, and the comparison has to be the same step as the
        // store. Deciding from a `fetch_min` and then storing under a separate lock is a
        // check-then-act race: with matches at 100 and 152, the thread at 152 reads `usize::MAX`
        // and decides to store, the thread at 100 reads 152 and also decides to store, and whichever
        // takes the lock last wins - which is 152 about half the time.
        //
        // So give every worker a slot of its own, keyed by `thread_index`. A slot has exactly one
        // writer, which is why the fold below needs no atomic, no lock, and no compare-exchange.
        // The caller then reduces the per-thread minima into the global one, sequentially, after the
        // join has already established happens-before.
        if self.iterator.is_empty() {
            return Ok(None);
        }
        let threads = self.pool.threads_count();
        // POLISH: a fresh Topology is probed here only because `ThreadPool` does not carry one;
        // it must be declared before `scratch` so the scratch allocation frees while it is alive.
        // A `&Topology` threaded through the parallel-iterator adapters would remove this.
        let topology = Topology::new().expect("failed to probe topology");
        let mut scratch: PinnedVec<BestCandidate<I::Item>> = PinnedVec::with_capacity_in(
            DomainAllocator::new(
                topology
                    .memory_domain_id_at_index(MemoryDomain(0))
                    .expect("in-range domain"),
            )
            .expect("failed to get allocator"),
            threads,
        )
        .expect("failed to allocate scratch");
        for _ in 0..threads {
            scratch.push(CacheAligned(None)).expect("failed to push");
        }

        self.reduce_with_scratch(
            scratch.as_mut_slice(),
            |slot, item, task, _at| {
                if !predicate(&item) {
                    return;
                }
                let index = task;
                match &slot.0 {
                    // ? Every task index is dispatched exactly once, so there are never ties
                    Some((best, _)) if *best <= index => {}
                    _ => slot.0 = Some((index, item)),
                }
            },
            |a, b| {
                let take_b = match (&a.0, &b.0) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some((a_index, _)), Some((b_index, _))) => b_index < a_index,
                };
                if take_b {
                    a.0 = b.0;
                }
            },
        )
        .map(|best| best.0.map(|(_, item)| item))
    }

    /// Searches for the last element that matches a predicate (deterministic, by index).
    ///
    /// Returns the element with the largest `task_index` among all matches.
    ///
    /// # Returns
    ///
    /// - `Some(item)` with the highest index if any match was found
    /// - `None` if no item matched or the iterator was empty
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = vec![10, 20, 30, 20, 10];
    ///
    /// let found = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .find_last(|&&x| x == 20)
    ///     .unwrap();
    ///
    /// assert_eq!(found, Some(&20)); // Index 3, not 1
    /// ```
    pub fn find_last<P>(self, predicate: P) -> Result<Option<I::Item>>
    where
        I::Item: Send,
        P: Fn(&I::Item) -> bool + Sync,
    {
        // The mirror of `find_first`, and it was racy for the same reason. A slot starts empty rather
        // than at a sentinel index, so index zero - a real index - cannot reject itself.
        if self.iterator.is_empty() {
            return Ok(None);
        }
        let threads = self.pool.threads_count();
        // POLISH: a fresh Topology is probed here only because `ThreadPool` does not carry one;
        // it must be declared before `scratch` so the scratch allocation frees while it is alive.
        // A `&Topology` threaded through the parallel-iterator adapters would remove this.
        let topology = Topology::new().expect("failed to probe topology");
        let mut scratch: PinnedVec<BestCandidate<I::Item>> = PinnedVec::with_capacity_in(
            DomainAllocator::new(
                topology
                    .memory_domain_id_at_index(MemoryDomain(0))
                    .expect("in-range domain"),
            )
            .expect("failed to get allocator"),
            threads,
        )
        .expect("failed to allocate scratch");
        for _ in 0..threads {
            scratch.push(CacheAligned(None)).expect("failed to push");
        }

        self.reduce_with_scratch(
            scratch.as_mut_slice(),
            |slot, item, task, _at| {
                if !predicate(&item) {
                    return;
                }
                let index = task;
                match &slot.0 {
                    Some((best, _)) if *best >= index => {}
                    _ => slot.0 = Some((index, item)),
                }
            },
            |a, b| {
                let take_b = match (&a.0, &b.0) {
                    (None, _) => true,
                    (Some(_), None) => false,
                    (Some((a_index, _)), Some((b_index, _))) => b_index > a_index,
                };
                if take_b {
                    a.0 = b.0;
                }
            },
        )
        .map(|best| best.0.map(|(_, item)| item))
    }

    /// Searches for any element that matches a predicate (non-deterministic).
    ///
    /// Uses cooperative cancellation: once a match is found, no further items are processed.
    /// If multiple items match, any one of them may be returned.
    ///
    /// # Returns
    ///
    /// - `Some(item)` if a matching item was found
    /// - `None` if no item matched or the iterator was empty
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..1000).collect();
    ///
    /// let found = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .find_any(|&&x| x == 42)
    ///     .unwrap();
    ///
    /// assert_eq!(found, Some(&42));
    /// ```
    pub fn find_any<P>(self, predicate: P) -> Result<Option<I::Item>>
    where
        I::Item: Send,
        P: Fn(&I::Item) -> bool + Sync,
    {
        use core::sync::atomic::{AtomicBool, Ordering};

        let ParallelRunner {
            pool,
            iterator,
            schedule,
        } = self;

        let stop = AtomicBool::new(false);
        let found = SyncOnceCell::new();
        iterator.drive(pool, schedule, &|item, _task, _at| {
            if stop.load(Ordering::Acquire) {
                return;
            }

            if predicate(&item) {
                let already_stopped = stop.swap(true, Ordering::Release);
                if !already_stopped {
                    // SAFETY: Only one thread sets stop to true, so only one write
                    unsafe { found.set(item) };
                }
            }
        })?;

        // SAFETY: All worker threads finished, exclusive access
        Ok(found.into_inner())
    }

    /// Returns `true` if any item matches the predicate.
    ///
    /// Stops searching after the first match is found.
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..1000).collect();
    ///
    /// let has_large = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .any(|&&x| x > 500)
    ///     .unwrap();
    ///
    /// assert!(has_large);
    /// ```
    pub fn any<P>(self, predicate: P) -> Result<bool>
    where
        I::Item: Send,
        P: Fn(&I::Item) -> bool + Sync,
    {
        self.find_any(predicate).map(|found| found.is_some())
    }

    /// Returns `true` if all items match the predicate.
    ///
    /// Stops searching after the first non-match is found.
    ///
    /// # Example
    ///
    /// ```
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..100).collect();
    ///
    /// let all_small = (&data[..])
    ///     .into_par_iter()
    ///     .with_pool(&mut pool)
    ///     .all(|&&x| x < 200)
    ///     .unwrap();
    ///
    /// assert!(all_small);
    /// ```
    pub fn all<P>(self, predicate: P) -> Result<bool>
    where
        I::Item: Send,
        P: Fn(&I::Item) -> bool + Sync,
    {
        self.any(|item| !predicate(item)).map(|found| !found)
    }

    /// Parallel reduction with a cache-aligned per-thread scratch accumulator.
    ///
    /// Allocates one `CacheAligned<T>` accumulator per thread in a node-local `PinnedVec`, so threads
    /// fold into local memory before the serial combine.
    ///
    /// Nearly identical to Rayon's reduce API, just requires explicit pool.
    ///
    /// # Arguments
    /// * `init` - Function to create initial accumulator value
    /// * `fold` - Function to accumulate items: `fn(&mut T, I::Item, usize, ThreadInDomain)`
    /// * `combine` - Function to merge two accumulators: `fn(T, T) -> T`
    ///
    /// # Example
    /// ```
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<u64> = (0..1000).collect();
    ///
    /// let total = (&data[..]).into_par_iter().with_pool(&mut pool)
    ///     .reduce(|| 0, |acc, value, _, _| *acc += *value, |a, b| a + b);
    /// ```
    pub fn reduce<T, Init, F, C>(self, init: Init, fold: F, combine: C) -> Result<T>
    where
        Init: Fn() -> T + Sync,
        T: Send + Sync + Default,
        F: Fn(&mut T, I::Item, usize, ThreadInDomain) + Sync,
        C: Fn(T, T) -> T,
    {
        // Handle empty iterators early
        if self.iterator.is_empty() {
            return Ok(init());
        }

        let threads = self.pool.threads_count();

        // Create cache-aligned scratch: one CacheAligned<T> per thread
        // Note: Using PinnedVec per compute_domain for true NUMA-awareness would be ideal,
        // but for simplicity we use a contiguous allocation here. The OS will still
        // tend to place this on the memory domain of the allocating thread.
        // POLISH: a fresh Topology is probed here only because `ThreadPool` does not carry one;
        // it must outlive `scratch`. A `&Topology` threaded through the reduce adapters removes this.
        let topology = Topology::new().expect("failed to probe topology");
        let mut scratch = PinnedVec::with_capacity_in(
            DomainAllocator::new(
                topology
                    .memory_domain_id_at_index(MemoryDomain(0))
                    .expect("in-range domain"),
            )
            .expect("failed to get allocator"),
            threads,
        )
        .expect("failed to allocate scratch");

        for _ in 0..threads {
            scratch.push(CacheAligned(init())).expect("failed to push");
        }

        // Fold phase uses reduce_with_scratch which indexes by thread_index
        self.reduce_with_scratch(
            scratch.as_mut_slice(),
            |acc, item, task, at| fold(&mut acc.0, item, task, at),
            |a, b| {
                let old_a = core::mem::take(&mut a.0);
                a.0 = combine(old_a, b.0);
            },
        )
        .map(|total| total.0)
    }

    /// Sum all items in parallel with NUMA-aware local accumulators.
    ///
    /// Works for owned values (usize, u64, etc.) and references (&u64, etc.).
    ///
    /// # Example
    /// ```
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data = vec![1u64, 2, 3, 4, 5];
    /// let sum: u64 = (&data[..]).into_par_iter().with_pool(&mut pool).sum().unwrap();
    /// assert_eq!(sum, 15);
    /// ```
    pub fn sum<T>(self) -> Result<T>
    where
        T: Send
            + Sync
            + Default
            + Copy
            + core::ops::AddAssign<I::Item>
            + core::ops::Add<Output = T>,
    {
        self.reduce(T::default, |acc, item, _, _| *acc += item, |a, b| a + b)
    }

    /// Count all items in parallel with NUMA-aware local counters.
    ///
    /// # Example
    /// ```
    /// use forkunion::*;
    /// let topology = Topology::new().unwrap();
    /// let mut pool = ThreadPool::spawn(&topology, 4).unwrap();
    /// let data: Vec<usize> = (0..1000).collect();
    /// let count = (&data[..]).into_par_iter().with_pool(&mut pool).count();
    /// ```
    pub fn count(self) -> Result<usize> {
        self.reduce(|| 0usize, |acc, _item, _, _| *acc += 1, |a, b| a + b)
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        let Map { base, mapper } = self;
        let mapped =
            move |item: I::Item, task: usize, at: ThreadInDomain| consumer(mapper(item), task, at);

        base.drive(pool, schedule, &mapped)
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        let Filter { base, predicate } = self;
        let filtered = move |item: I::Item, task: usize, at: ThreadInDomain| {
            if predicate(&item) {
                consumer(item, task, at);
            }
        };

        base.drive(pool, schedule, &filtered)
    }
}

#[derive(Clone, Copy)]
pub struct ParallelSlice<'a, T> {
    data: &'a [T],
}

impl<'a, T> ParallelSlice<'a, T> {
    #[must_use]
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    pub fn for_each_static<F>(self, pool: &mut ThreadPool, function: F) -> Result<()>
    where
        T: Sync,
        F: Fn(&'a T, usize, ThreadInDomain) + Sync,
    {
        self.drive_static(pool, function)
    }

    pub fn for_each_dynamic<F>(self, pool: &mut ThreadPool, function: F) -> Result<()>
    where
        T: Sync,
        F: Fn(&'a T, usize, ThreadInDomain) + Sync,
    {
        self.drive_dynamic(pool, function)
    }

    #[must_use]
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        if self.data.is_empty() {
            return Ok(());
        }

        let slice = self.data;
        schedule.dispatch(pool, slice.len(), move |task, at| {
            let item = unsafe { slice.get_unchecked(task) };
            consumer(item, task, at);
        })
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        let len = self.left.data.len();
        assert_eq!(len, self.right.data.len(), "zip requires equal lengths");
        if len == 0 {
            return Ok(());
        }

        let left = self.left.data;
        let right = self.right.data;
        schedule.dispatch(pool, len, move |task, at| {
            let lhs = unsafe { left.get_unchecked(task) };
            let rhs = unsafe { right.get_unchecked(task) };
            consumer((lhs, rhs), task, at);
        })
    }
}

pub struct ParallelSliceMut<'a, T> {
    ptr: SyncMutPtr<T>,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T> ParallelSliceMut<'a, T> {
    #[must_use]
    pub fn new(data: &'a mut [T]) -> Self {
        Self {
            ptr: SyncMutPtr::new(data.as_mut_ptr()),
            len: data.len(),
            _marker: PhantomData,
        }
    }

    pub fn for_each_static<F>(self, pool: &mut ThreadPool, function: F) -> Result<()>
    where
        T: Send,
        F: Fn(&'a mut T, usize, ThreadInDomain) + Sync,
    {
        self.drive_static(pool, function)
    }

    pub fn for_each_dynamic<F>(self, pool: &mut ThreadPool, function: F) -> Result<()>
    where
        T: Send,
        F: Fn(&'a mut T, usize, ThreadInDomain) + Sync,
    {
        self.drive_dynamic(pool, function)
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        if self.len == 0 {
            return Ok(());
        }

        let ptr = self.ptr;
        schedule.dispatch(pool, self.len, move |task, at| {
            let item = unsafe { &mut *ptr.get(task) };
            consumer(item, task, at);
        })
    }
}

#[derive(Clone)]
pub struct ParallelRange {
    range: core::ops::Range<usize>,
}

impl ParallelRange {
    #[must_use]
    pub fn new(range: core::ops::Range<usize>) -> Self {
        Self { range }
    }
}

impl ParallelIterator for ParallelRange {
    type Item = usize;

    fn len(&self) -> usize {
        self.range.len()
    }

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        let len = self.range.len();
        if len == 0 {
            return Ok(());
        }

        let start = self.range.start;
        schedule.dispatch(pool, len, move |task, at| {
            let index = start + task;
            consumer(index, index, at);
        })
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
    #[must_use]
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

    fn drive<S, F>(self, pool: &mut ThreadPool, schedule: S, consumer: &F) -> Result<()>
    where
        S: ParallelSchedule,
        F: Fn(Self::Item, usize, ThreadInDomain) + Sync,
    {
        let ParallelExactIter { len, indexer, .. } = self;
        if len == 0 {
            return Ok(());
        }

        schedule.dispatch_slices(pool, len, move |range, at| {
            for task in range {
                consumer(indexer(task), task, at);
            }
        })
    }
}

pub mod prelude {
    pub use super::{
        DynamicScheduler, IntoParallelIterator, ParallelIterator, ParallelIteratorExt,
        ParallelRunner, StaticScheduler,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::tests::hw_threads;
    use crate::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::vec;
    use std::vec::Vec;

    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_slice_static_for_each() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<usize> = (0..512).collect();
        let total = AtomicUsize::new(0);

        (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each(|value, _task, _at| {
                total.fetch_add(*value, Ordering::Relaxed);
            })
            .unwrap();

        assert_eq!(total.load(Ordering::Relaxed), data.iter().sum());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_slice_mut_dynamic_for_each() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let mut data: Vec<usize> = (0..256).collect();

        (&mut data[..])
            .into_par_iter()
            .with_schedule(&mut pool, DynamicScheduler)
            .for_each(|value, _task, _at| {
                *value *= 2;
            })
            .unwrap();

        for (i, v) in data.iter().enumerate() {
            assert_eq!(*v, i * 2);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_slice_zip_sum() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let a: Vec<usize> = (0..128).collect();
        let b: Vec<usize> = (0..128).rev().collect();
        let sums: Arc<Vec<AtomicUsize>> =
            Arc::new((0..hw_threads()).map(|_| AtomicUsize::new(0)).collect());
        let shared = Arc::clone(&sums);

        (&a[..])
            .into_par_iter()
            .zip((&b[..]).into_par_iter())
            .with_pool(&mut pool)
            .for_each(|(lhs, rhs), _task, at| {
                shared[at.thread % shared.len()].fetch_add(lhs + rhs, Ordering::Relaxed);
            })
            .unwrap();

        let total: usize = sums.iter().map(|v| v.load(Ordering::Relaxed)).sum();
        let expected: usize = a.iter().zip(b.iter()).map(|(x, y)| x + y).sum();
        assert_eq!(total, expected);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn parallel_exact_iter_dispatch() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let mut values = vec![0usize; 256];
        let ptr = SyncMutPtr::new(values.as_mut_ptr());
        (0..values.len())
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each(|index, task, _at| {
                let slot = unsafe { &mut *ptr.get(task) };
                *slot = index * index;
            })
            .unwrap();

        for (idx, val) in values.iter().enumerate() {
            assert_eq!(*val, idx * idx);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn scratch_reduction_collects_sum() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<usize> = (0..1024).collect();
        let mut scratch = vec![0usize; pool.threads_count()];

        (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .fold_with_scratch(scratch.as_mut_slice(), |slot, value, _, _| {
                *slot += *value;
            })
            .unwrap();

        let total: usize = scratch.iter().sum();
        let expected: usize = data.iter().sum();
        assert_eq!(total, expected);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_with_scratch_sum() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1024).collect();
        let mut scratch: Vec<CacheAligned<u64>> =
            (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();

        let total = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .reduce_with_scratch(
                scratch.as_mut_slice(),
                |acc, value, _, _| acc.0 += *value,
                |a, b| a.0 += b.0,
            )
            .unwrap();

        assert_eq!(total.0, data.iter().sum());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_with_scratch_dynamic() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<usize> = (0..1000).collect();
        let mut scratch: Vec<CacheAligned<usize>> =
            (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();

        let total = (&data[..])
            .into_par_iter()
            .with_schedule(&mut pool, DynamicScheduler)
            .reduce_with_scratch(
                scratch.as_mut_slice(),
                |a, v, _, _| a.0 += *v,
                |x, y| x.0 += y.0,
            )
            .unwrap();

        assert_eq!(total.0, data.iter().sum());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_sum() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (1..=1000).collect();
        let total: u64 = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .sum()
            .unwrap();
        assert_eq!(total, data.iter().sum());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_count() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<usize> = (0..1000).collect();
        let count = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .count()
            .unwrap();
        assert_eq!(count, 1000);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_product() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data = [2u64, 3, 5, 7];
        let product = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .reduce(|| 1u64, |a, v, _, _| *a *= *v, |x, y| x * y)
            .unwrap();
        assert_eq!(product, data.iter().product());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_empty() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = vec![];
        assert_eq!(
            (&data[..])
                .into_par_iter()
                .with_pool(&mut pool)
                .sum::<u64>()
                .unwrap(),
            0
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn reduce_range() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let total: usize = (0..10_000)
            .into_par_iter()
            .with_pool(&mut pool)
            .sum()
            .unwrap();
        assert_eq!(total, (0..10_000).sum::<usize>());
    }

    // Early-exit API tests

    #[cfg_attr(miri, ignore)]
    #[test]
    fn try_for_each_success() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each_fallible(|&x, _, _| if x < 1000 { Ok(()) } else { Err("too large") })
            .unwrap();
        assert!(result.is_ok());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn try_for_each_early_exit() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each_fallible(|&x, _, _| if x < 500 { Ok(()) } else { Err(x) })
            .unwrap();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!((500..1000).contains(&err));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn try_for_each_empty() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = vec![];
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .for_each_fallible(|&_x, _, _| -> core::result::Result<(), &str> {
                Err("should not run")
            })
            .unwrap();
        assert!(result.is_ok());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn try_fold_with_scratch_success() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let mut scratch: Vec<CacheAligned<u64>> =
            (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();

        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .fold_with_scratch_fallible(scratch.as_mut_slice(), |acc, &value, _, _| {
                acc.0 += value;
                Ok::<(), &str>(())
            })
            .unwrap();

        assert!(result.is_ok());
        let total: u64 = scratch.iter().map(|x| x.0).sum();
        assert_eq!(total, data.iter().sum());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn try_fold_with_scratch_early_exit() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let mut scratch: Vec<CacheAligned<u64>> =
            (0..pool.threads_count()).map(|_| CacheAligned(0)).collect();

        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .fold_with_scratch_fallible(scratch.as_mut_slice(), |acc, &value, _, _| {
                if value >= 500 {
                    Err(value)
                } else {
                    acc.0 += value;
                    Ok(())
                }
            })
            .unwrap();

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!((500..1000).contains(&err));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_first_deterministic() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        // Find first even number >= 100
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_first(|&&x| x >= 100 && x % 2 == 0)
            .unwrap();
        assert_eq!(found, Some(&100));
    }

    /// Repeated, because a lost race shows up as a flake rather than a failure.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_first_not_found() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..100).collect();
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_first(|&&x| x >= 200)
            .unwrap();
        assert_eq!(found, None);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_first_last_deterministic_under_contention() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..10_000).collect();
        for _ in 0..64 {
            let first = (&data[..])
                .into_par_iter()
                .with_pool(&mut pool)
                .find_first(|&&x| x >= 100 && x % 2 == 0)
                .unwrap();
            assert_eq!(
                first,
                Some(&100),
                "find_first must return the lowest matching index"
            );

            let last = (&data[..])
                .into_par_iter()
                .with_pool(&mut pool)
                .find_last(|&&x| x <= 9_000 && x % 2 == 0)
                .unwrap();
            assert_eq!(
                last,
                Some(&9_000),
                "find_last must return the highest matching index"
            );
        }
    }

    /// Index zero is a real index: `find_last` must not reject it against an empty slot.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_last_accepts_index_zero() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1_000).collect();
        let last = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_last(|&&x| x == 0)
            .unwrap();
        assert_eq!(last, Some(&0));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_last_deterministic() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        // Find last even number < 900
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_last(|&&x| x < 900 && x % 2 == 0)
            .unwrap();
        assert_eq!(found, Some(&898));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_last_not_found() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..100).collect();
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_last(|&&x| x >= 200)
            .unwrap();
        assert_eq!(found, None);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_any_found() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_any(|&&x| x == 42)
            .unwrap();
        assert_eq!(found, Some(&42));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_any_not_found() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_any(|&&x| x == 2000)
            .unwrap();
        assert_eq!(found, None);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn find_any_empty() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = vec![];
        let found = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .find_any(|&&_x| true)
            .unwrap();
        assert_eq!(found, None);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn any_true() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .any(|&&x| x == 42)
            .unwrap();
        assert!(result);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn any_false() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .any(|&&x| x >= 2000)
            .unwrap();
        assert!(!result);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn any_empty() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = vec![];
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .any(|&&_x| true)
            .unwrap();
        assert!(!result);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn all_true() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .all(|&&x| x < 2000)
            .unwrap();
        assert!(result);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn all_false() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = (0..1000).collect();
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .all(|&&x| x < 500)
            .unwrap();
        assert!(!result);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn all_empty() {
        let topology = Topology::new().unwrap();
        let mut pool = spawn(&topology, hw_threads());
        let data: Vec<u64> = vec![];
        let result = (&data[..])
            .into_par_iter()
            .with_pool(&mut pool)
            .all(|&&_x| false)
            .unwrap();
        assert!(result); // vacuous truth
    }
}
