//! NUMA-aware allocation - domain allocators, pinned vectors, replicated and sharded arrays.
//!
//! Owns the `fu_allocate_*`/`fu_free_*` FFI; mirrors the C++ `allocators` header.

use crate::parallel::{ParallelSlice, ParallelSliceMut};
use crate::topology::{MemoryDomain, MemoryDomainId, Topology};
use crate::types::{SyncMutPtr, DEFAULT_ALIGNMENT};
use core::ffi::c_void;
use core::ptr::NonNull;
use core::slice;

// C FFI declarations
extern "C" {
    fn fu_allocate_at_least_on_domain_id(
        memory_domain_id: i32,
        minimum_bytes: usize,
        allocated_bytes: *mut usize,
        bytes_per_page: *mut usize,
    ) -> *mut c_void;
    fn fu_allocate_on_domain_id(memory_domain_id: i32, bytes: usize) -> *mut c_void;
    fn fu_free_on_domain_id(memory_domain_id: i32, pointer: *mut c_void, bytes: usize);
    fn fu_allocate_symmetric(
        topology: *mut c_void,
        bytes_per_domain: usize,
        stride_bytes: *mut usize,
        memory_domains_count: *mut usize,
        total_bytes: *mut usize,
        bytes_per_page: *mut usize,
    ) -> *mut c_void;
    fn fu_free_symmetric(base: *mut c_void, total_bytes: usize);
}

/// Result of a memory-domain allocation; carries only the OS id, so it can outlive the topology.
#[derive(Debug)]
pub struct AllocationResult {
    ptr: NonNull<u8>,
    allocated_bytes: usize,
    bytes_per_page: usize,
    // The OS memory-domain id is all `fu_free_on_domain_id` needs, so the allocation carries no topology handle
    // and can outlive the `Topology` it came from.
    memory_domain_id: MemoryDomainId,
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

    /// Returns the OS id of the memory domain this memory was allocated on.
    pub fn memory_domain_id(&self) -> MemoryDomainId {
        self.memory_domain_id
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
            fu_free_on_domain_id(
                self.memory_domain_id.get(),
                self.ptr.as_ptr() as *mut c_void,
                self.allocated_bytes,
            );
        }
    }
}

// Safety: AllocationResult can be sent between threads since it owns its memory

unsafe impl Send for AllocationResult {}
// Safety: AllocationResult can be shared between threads with proper synchronization
unsafe impl Sync for AllocationResult {}

/// An allocator bound to a single memory domain, so every block it hands out is domain-local.
///
/// On a machine whose memory access latency varies with the domain a block lives on, keeping an
/// allocation on the domain of the cores that read it avoids the cross-domain penalty.
///
/// # Examples
///
/// ```rust
/// use forkunion::*;
/// let topology = Topology::new().unwrap();
/// let id = topology.memory_domain_id_at_index(MemoryDomain(0));
/// let allocator = DomainAllocator::new(id).expect("failed to bind allocator to memory domain 0");
/// let allocation = allocator.allocate(1024).expect("failed to allocate 1024 bytes");
///
/// // Access the allocated memory
/// let memory_slice = allocation.as_slice();
/// assert_eq!(memory_slice.len(), 1024);
/// println!("Allocated {} bytes on memory domain {}",
///          allocation.allocated_bytes(), allocation.memory_domain_id().get());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DomainAllocator {
    memory_domain_id: MemoryDomainId,
}

impl DomainAllocator {
    /// Creates an allocator bound to the memory domain named by `memory_domain_id`.
    ///
    /// # Arguments
    ///
    /// * `memory_domain_id` - The OS memory-domain id from [`Topology::memory_domain_id_at_index`]
    ///
    /// # Errors
    ///
    /// Returns `None` if the id is the `-1` sentinel, i.e. it names no domain.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// // Bind an allocator to the first memory domain
    /// let id = topology.memory_domain_id_at_index(MemoryDomain(0));
    /// let allocator = DomainAllocator::new(id).expect("memory domain 0 should be available");
    ///
    /// // Bind another to a second memory domain when the machine has one
    /// let memory_domains = topology.memory_domains_count();
    /// if memory_domains > 1 {
    ///     let id2 = topology.memory_domain_id_at_index(MemoryDomain(1));
    ///     let allocator2 = DomainAllocator::new(id2).expect("memory domain 1 should be available");
    ///     println!("Bound allocator to memory domain: {}", allocator2.memory_domain_id().get());
    /// }
    /// ```
    pub fn new(memory_domain_id: MemoryDomainId) -> Option<Self> {
        if !memory_domain_id.is_valid() {
            return None;
        }

        Some(Self { memory_domain_id })
    }

    /// Returns the OS id of the memory domain this allocator is bound to.
    pub fn memory_domain_id(&self) -> MemoryDomainId {
        self.memory_domain_id
    }

    /// Allocates memory with at least the requested size on this allocator's memory domain.
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).unwrap();
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
            let ptr = fu_allocate_at_least_on_domain_id(
                self.memory_domain_id.get(),
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
                memory_domain_id: self.memory_domain_id,
            })
        }
    }

    /// Allocates exactly the requested number of bytes on this allocator's memory domain.
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let id = topology.memory_domain_id_at_index(MemoryDomain(0));
    /// let allocator = DomainAllocator::new(id).unwrap();
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
            let ptr = fu_allocate_on_domain_id(self.memory_domain_id.get(), bytes);

            if ptr.is_null() {
                return None;
            }

            Some(AllocationResult {
                ptr: NonNull::new_unchecked(ptr as *mut u8),
                allocated_bytes: bytes,
                bytes_per_page: 0, // Not provided by fu_allocate
                memory_domain_id: self.memory_domain_id,
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).unwrap();
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
        // The allocator guarantees `DEFAULT_ALIGNMENT`, so any element type up to that alignment is
        // satisfied directly; a stricter type cannot be honored and is refused.
        if core::mem::align_of::<T>() > DEFAULT_ALIGNMENT {
            return None;
        }
        let bytes = count.checked_mul(core::mem::size_of::<T>())?;
        self.allocate(bytes)
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).unwrap();
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

/// Creates an allocator for the first memory domain, index 0, when the specific domain does not matter.
///
/// # Examples
///
/// ```rust
/// use forkunion::*;
///
/// let topology = Topology::new().unwrap();
/// let allocator = default_pinned_allocator(&topology).expect("No memory domains available");
/// let allocation = allocator.allocate(1024).expect("Failed to allocate");
///
/// // The default allocator uses the first memory domain
/// assert_eq!(allocation.memory_domain_id(), topology.memory_domain_id_at_index(MemoryDomain(0)));
///
/// // For more control, pin to a specific domain
/// let domains = topology.memory_domains_count();
/// println!("System has {} memory domains available", domains);
///
/// if domains > 1 {
///     let allocator_domain1 = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(1))).expect("domain 1 available");
///     let allocation2 = allocator_domain1.allocate(2048).expect("Failed to allocate on domain 1");
///     assert_eq!(allocation2.memory_domain_id(), topology.memory_domain_id_at_index(MemoryDomain(1)));
/// }
/// ```
pub fn default_pinned_allocator(topology: &Topology) -> Option<DomainAllocator> {
    DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
}

/// A Vec-like container that uses NUMA-aware pinned memory allocation.
///
/// `PinnedVec<T>` provides a dynamic array that allocates memory on a specific
/// memory domain, which should correspond to a `compute_domain_index` for optimal
/// performance with `ThreadPool`. It automatically manages growth and shrinkage.
///
/// # Examples
///
/// ```rust
/// use forkunion::*;
///
/// // Create a vector on memory domain 0
/// let topology = Topology::new().unwrap();
/// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    allocator: DomainAllocator,
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
    /// * `allocator` - The `DomainAllocator` to use for memory allocation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
    /// let vec = PinnedVec::<i32>::new_in(allocator);
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 0);
    /// ```
    pub fn new_in(allocator: DomainAllocator) -> Self {
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
    /// * `allocator` - The `DomainAllocator` to use for memory allocation
    /// * `capacity` - The initial capacity to allocate
    ///
    /// # Errors
    ///
    /// Returns `None` if allocation fails.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
    /// let vec = PinnedVec::<i32>::with_capacity_in(allocator, 100).expect("Failed to create vec");
    /// assert_eq!(vec.len(), 0);
    /// assert_eq!(vec.capacity(), 100);
    /// ```
    pub fn with_capacity_in(allocator: DomainAllocator, capacity: usize) -> Option<Self> {
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

    /// Returns the OS id of the memory domain this vector's memory is allocated on.
    pub fn memory_domain_id(&self) -> MemoryDomainId {
        self.allocator.memory_domain_id()
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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
    /// use forkunion::*;
    ///
    /// let topology = Topology::new().unwrap();
    /// let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0))).expect("Failed to create alloc");
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

/// One symmetric mapping the C allocator stripes across every memory domain and frees as a whole.
///
/// Slice `d` starts at `base + d * stride_bytes` and is bound to its own memory domain. The mapping owns
/// its storage and unmaps it on drop; like [`AllocationResult`] it carries no topology handle.
struct SymmetricAllocation {
    base: NonNull<u8>,
    stride_bytes: usize,
    domains: usize,
    total_bytes: usize,
}

impl SymmetricAllocation {
    /// Allocates one mapping of `bytes_per_domain` per memory domain, or `None` on failure.
    fn new(topology: &Topology, bytes_per_domain: usize) -> Option<Self> {
        let mut stride_bytes = 0usize;
        let mut domains = 0usize;
        let mut total_bytes = 0usize;
        let base = unsafe {
            fu_allocate_symmetric(
                topology.raw(),
                bytes_per_domain,
                &mut stride_bytes,
                &mut domains,
                &mut total_bytes,
                core::ptr::null_mut(),
            )
        };
        let base = NonNull::new(base as *mut u8)?;
        Some(Self {
            base,
            stride_bytes,
            domains,
            total_bytes,
        })
    }

    /// Base of the slice on `memory_domain`, at `base + memory_domain * stride_bytes`.
    fn slice_base(&self, memory_domain: usize) -> *mut u8 {
        unsafe { self.base.as_ptr().add(memory_domain * self.stride_bytes) }
    }
}

impl Drop for SymmetricAllocation {
    fn drop(&mut self) {
        unsafe { fu_free_symmetric(self.base.as_ptr() as *mut c_void, self.total_bytes) };
    }
}

unsafe impl Send for SymmetricAllocation {}
unsafe impl Sync for SymmetricAllocation {}

/// One full length-`n` copy of a sequence per memory domain, so every thread reads a node-local replica.
///
/// A thin owner of one symmetric mapping - replica `d` lives at `base + d * stride_bytes()` and holds `n`
/// elements. Raw uninitialized storage: the caller fills every replica and keeps them coherent, mirroring
/// the C++ `replicated_array`. `T` must be plain-old-data - any bit pattern is a valid value - since the
/// container runs no constructors or destructors.
pub struct ReplicatedArray<T> {
    allocation: Option<SymmetricAllocation>,
    len: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: Copy> ReplicatedArray<T> {
    /// An empty array holding no mapping.
    pub const fn new() -> Self {
        Self {
            allocation: None,
            len: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Allocates one uninitialized length-`n` replica per memory domain; the caller first-touches them.
    pub fn try_new(topology: &Topology, n: usize) -> Option<Self> {
        if n == 0 {
            return Some(Self::new());
        }
        let allocation = SymmetricAllocation::new(topology, n * core::mem::size_of::<T>())?;
        Some(Self {
            allocation: Some(allocation),
            len: n,
            _phantom: core::marker::PhantomData,
        })
    }

    /// The logical length of each replica.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of per-domain replicas.
    pub fn memory_domains_count(&self) -> usize {
        self.allocation
            .as_ref()
            .map_or(0, |allocation| allocation.domains)
    }

    /// The page-aligned byte distance between consecutive replicas.
    pub fn stride_bytes(&self) -> usize {
        self.allocation
            .as_ref()
            .map_or(0, |allocation| allocation.stride_bytes)
    }

    /// Raw start of the replica on `memory_domain`, for concurrent first-touch fills through a raw pointer.
    pub fn replica_ptr(&self, memory_domain: MemoryDomain) -> *mut T {
        let allocation = self.allocation.as_ref().expect("empty ReplicatedArray");
        allocation.slice_base(memory_domain.get()) as *mut T
    }

    /// The whole replica living on `memory_domain`.
    pub fn on_memory_domain(&self, memory_domain: MemoryDomain) -> &[T] {
        unsafe { slice::from_raw_parts(self.replica_ptr(memory_domain), self.len) }
    }

    /// The whole replica living on `memory_domain`, mutably.
    pub fn on_memory_domain_mut(&mut self, memory_domain: MemoryDomain) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.replica_ptr(memory_domain), self.len) }
    }

    /// One element of the replica on `memory_domain`.
    pub fn at(&self, memory_domain: MemoryDomain, local_index: usize) -> &T {
        &self.on_memory_domain(memory_domain)[local_index]
    }
}

impl<T: Copy> Default for ReplicatedArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for ReplicatedArray<T> {}
unsafe impl<T: Sync> Sync for ReplicatedArray<T> {}

/// Where a logical element lives - which memory domain, and its index within that shard.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ShardLocation {
    pub memory_domain: MemoryDomain,
    pub local_index: usize,
}

/// A sequence partitioned across memory domains as contiguous segments, each element stored once.
///
/// Each domain owns a contiguous logical segment of `segment()` elements, so element `i` lives at
/// `{i / segment(), i % segment()}` - see [`location_of`](Self::location_of) - and a scan of a domain's
/// shard is sequential in memory. Backed by one symmetric mapping; the trailing shard may be short. `T`
/// must be plain-old-data, mirroring [`ReplicatedArray`].
pub struct ShardedArray<T> {
    allocation: Option<SymmetricAllocation>,
    len: usize,
    segment: usize,
    _phantom: core::marker::PhantomData<T>,
}

impl<T: Copy> ShardedArray<T> {
    /// An empty array holding no mapping.
    pub const fn new() -> Self {
        Self {
            allocation: None,
            len: 0,
            segment: 0,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Allocates uninitialized storage for `n` elements partitioned round-robin across the domains.
    pub fn try_new(topology: &Topology, n: usize) -> Option<Self> {
        if n == 0 {
            return Some(Self::new());
        }
        let domains = topology.memory_domains_count();
        if domains == 0 {
            return None;
        }
        let segment = n.div_ceil(domains);
        let allocation = SymmetricAllocation::new(topology, segment * core::mem::size_of::<T>())?;
        Some(Self {
            allocation: Some(allocation),
            len: n,
            segment,
            _phantom: core::marker::PhantomData,
        })
    }

    /// The logical length, summed across the shards.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the array holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The number of shards, one per memory domain.
    pub fn memory_domains_count(&self) -> usize {
        self.allocation
            .as_ref()
            .map_or(0, |allocation| allocation.domains)
    }

    /// The page-aligned byte distance between consecutive shards.
    pub fn stride_bytes(&self) -> usize {
        self.allocation
            .as_ref()
            .map_or(0, |allocation| allocation.stride_bytes)
    }

    /// The contiguous logical segment size per domain - `ceil(n / memory_domains_count())`.
    pub fn segment(&self) -> usize {
        self.segment
    }

    /// How many elements the shard on `memory_domain` holds - a trailing shard may be shorter.
    pub fn length_on_memory_domain(&self, memory_domain: MemoryDomain) -> usize {
        let start = memory_domain.get() * self.segment;
        if start >= self.len {
            0
        } else {
            (self.len - start).min(self.segment)
        }
    }

    /// The memory domain and local index that store logical element `logical_index`.
    pub fn location_of(&self, logical_index: usize) -> ShardLocation {
        ShardLocation {
            memory_domain: MemoryDomain(logical_index / self.segment),
            local_index: logical_index % self.segment,
        }
    }

    /// The logical index of the element at `local_index` on `memory_domain` - inverse of `location_of`.
    pub fn logical_index_of(&self, memory_domain: MemoryDomain, local_index: usize) -> usize {
        memory_domain.get() * self.segment + local_index
    }

    /// Raw start of the shard on `memory_domain`, for concurrent fills through a raw pointer.
    pub fn shard_ptr(&self, memory_domain: MemoryDomain) -> *mut T {
        let allocation = self.allocation.as_ref().expect("empty ShardedArray");
        allocation.slice_base(memory_domain.get()) as *mut T
    }

    /// The whole shard living on `memory_domain`.
    pub fn on_memory_domain(&self, memory_domain: MemoryDomain) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.shard_ptr(memory_domain),
                self.length_on_memory_domain(memory_domain),
            )
        }
    }

    /// The whole shard living on `memory_domain`, mutably.
    pub fn on_memory_domain_mut(&mut self, memory_domain: MemoryDomain) -> &mut [T] {
        let len = self.length_on_memory_domain(memory_domain);
        unsafe { slice::from_raw_parts_mut(self.shard_ptr(memory_domain), len) }
    }

    /// The single home of a logical element.
    pub fn at(&self, memory_domain: MemoryDomain, local_index: usize) -> &T {
        &self.on_memory_domain(memory_domain)[local_index]
    }
}

impl<T: Copy> Default for ShardedArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl<T: Send> Send for ShardedArray<T> {}
unsafe impl<T: Sync> Sync for ShardedArray<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::*;

    use std::vec::Vec;

    /// Each replica is an independent length-`n` buffer, so a domain-dependent fill must read back
    /// exactly on its own domain - any aliasing between replicas would corrupt the pattern.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn replicated_array_per_domain_buffer() {
        let topology = Topology::new().unwrap();
        let n = 4096usize;
        let mut replicas: ReplicatedArray<u32> =
            ReplicatedArray::try_new(&topology, n).expect("replicated array");
        assert_eq!(replicas.len(), n);
        assert_eq!(
            replicas.memory_domains_count(),
            topology.memory_domains_count()
        );

        let domains = replicas.memory_domains_count();
        for domain in 0..domains {
            let replica = replicas.on_memory_domain_mut(MemoryDomain(domain));
            assert_eq!(replica.len(), n);
            for (index, slot) in replica.iter_mut().enumerate() {
                *slot = (domain * n + index) as u32;
            }
        }

        for domain in 0..domains {
            for index in 0..n {
                assert_eq!(
                    *replicas.at(MemoryDomain(domain), index),
                    (domain * n + index) as u32
                );
            }
        }
    }

    /// The shards tile the logical range exactly once, and `location_of` is the inverse of
    /// `logical_index_of` - every element has a single home and the two mappings agree.
    #[cfg_attr(miri, ignore)]
    #[test]
    fn sharded_array_segment_round_trip() {
        let topology = Topology::new().unwrap();
        let n = 4096usize;
        let mut shards: ShardedArray<u32> =
            ShardedArray::try_new(&topology, n).expect("sharded array");
        let domains = shards.memory_domains_count();
        let segment = shards.segment();
        assert_eq!(shards.len(), n);

        let footprint: usize = (0..domains)
            .map(|domain| shards.length_on_memory_domain(MemoryDomain(domain)))
            .sum();
        assert_eq!(footprint, n);

        for domain in 0..domains {
            let shard = shards.on_memory_domain_mut(MemoryDomain(domain));
            for (local, slot) in shard.iter_mut().enumerate() {
                *slot = (domain * segment + local) as u32;
            }
        }

        for logical in 0..n {
            let location = shards.location_of(logical);
            assert_eq!(
                shards.logical_index_of(location.memory_domain, location.local_index),
                logical
            );
            assert_eq!(
                *shards.at(location.memory_domain, location.local_index),
                logical as u32
            );
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_allocator_creation() {
        let topology = Topology::new().unwrap();
        let memory_domains = topology.memory_domains_count();
        assert!(
            memory_domains > 0,
            "system should have at least one memory domain"
        );

        // Valid memory domain
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("memory domain 0 should be available");
        assert_eq!(allocator.memory_domain_id().get(), 0);

        // Invalid memory domain
        let invalid_allocator = DomainAllocator::new(
            topology.memory_domain_id_at_index(MemoryDomain(memory_domains + 10)),
        );
        assert!(
            invalid_allocator.is_none(),
            "an invalid memory domain should return None"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn basic_allocation() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let allocation = allocator
            .allocate(1024)
            .expect("Failed to allocate 1024 bytes");

        assert_eq!(allocation.allocated_bytes(), 1024);
        assert_eq!(allocation.memory_domain_id().get(), 0);

        // Test that we can write to the memory
        let slice = allocation.as_slice();
        assert_eq!(slice.len(), 1024);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn allocate_zero_bytes() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let allocation = allocator.allocate(0);
        assert!(
            allocation.is_none(),
            "Allocating 0 bytes should return None"
        );
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn allocate_at_least() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let allocation = allocator
            .allocate_at_least(1000)
            .expect("Failed to allocate at least 1000 bytes");

        assert!(allocation.allocated_bytes() >= 1000);
        assert_eq!(allocation.memory_domain_id().get(), 0);

        // bytes_per_page should be set to something reasonable
        if allocation.bytes_per_page() > 0 {
            assert!(allocation.bytes_per_page() >= 512); // Reasonable minimum page size
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_creation() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let vec = PinnedVec::<i32>::new_in(allocator);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 0);
        assert_eq!(vec.memory_domain_id().get(), 0);
        assert!(vec.is_empty());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_with_capacity() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let vec = PinnedVec::<i32>::with_capacity_in(allocator, 10).expect("Failed to create vec");
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 10);
        assert_eq!(vec.memory_domain_id().get(), 0);
        assert!(vec.is_empty());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_push_pop() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_indexing() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_clear() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        vec.push(1).expect("Failed to push");
        vec.push(2).expect("Failed to push");
        vec.push(3).expect("Failed to push");

        assert_eq!(vec.len(), 3);
        vec.clear();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_insert_remove() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_reserve() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_extend_from_slice() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
        let mut vec = PinnedVec::<i32>::new_in(allocator);
        let data = [1, 2, 3, 4, 5];

        vec.extend_from_slice(&data).expect("Failed to extend");
        assert_eq!(vec.len(), 5);
        for (i, &value) in data.iter().enumerate() {
            assert_eq!(vec[i], value);
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_iterators() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_slices() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_growth() {
        let topology = Topology::new().unwrap();
        let allocator = DomainAllocator::new(topology.memory_domain_id_at_index(MemoryDomain(0)))
            .expect("Failed to create alloc");
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

    #[cfg_attr(miri, ignore)]
    #[test]
    fn pinned_vec_invalid_memory_domain() {
        let topology = Topology::new().unwrap();
        let memory_domains = topology.memory_domains_count();
        let allocator = DomainAllocator::new(
            topology.memory_domain_id_at_index(MemoryDomain(memory_domains + 1)),
        );
        assert!(allocator.is_none());
    }

    #[test]
    fn pinned_vec_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<PinnedVec<i32>>();
        assert_sync::<PinnedVec<i32>>();
    }
}
