//! Machine topology, capabilities, and library version - the read-only view of the hardware.
//!
//! Owns the `fu_topology_*`, capability, and version FFI; mirrors the C++ `topology` header.

use core::ffi::{c_char, c_int, c_void};

extern "C" {
    fn fu_version_major() -> c_int;
    fn fu_version_minor() -> c_int;
    fn fu_version_patch() -> c_int;
    fn fu_comptime_capabilities() -> u32;
    fn fu_runtime_capabilities() -> u32;
    fn fu_name_capabilities(caps: u32, buf: *mut c_char, len: usize) -> usize;

    fn fu_topology_new() -> *mut c_void;
    fn fu_topology_delete(topology: *mut c_void);

    fn fu_logical_cores_count_in(topology: *mut c_void, compute_domain_index: usize) -> usize;
    fn fu_logical_cores_count(topology: *mut c_void) -> usize;
    fn fu_compute_domains_count(topology: *mut c_void) -> usize;
    fn fu_compute_level_in(topology: *mut c_void, compute_domain_index: usize) -> usize;
    fn fu_compute_levels_count(topology: *mut c_void) -> usize;
    fn fu_compute_capacity_in(topology: *mut c_void, compute_domain_index: usize) -> usize;
    fn fu_compute_cache_bytes_in(topology: *mut c_void, compute_domain_index: usize) -> usize;

    fn fu_memory_domains_count(topology: *mut c_void) -> usize;
    fn fu_local_memory_of(topology: *mut c_void, compute_domain_index: usize) -> usize;
    fn fu_volume_ram_in(topology: *mut c_void, memory_domain_index: usize) -> usize;
    fn fu_volume_ram(topology: *mut c_void) -> usize;
    fn fu_volume_huge_pages_in(topology: *mut c_void, memory_domain_index: usize) -> usize;
    fn fu_volume_huge_pages(topology: *mut c_void) -> usize;
    fn fu_huge_pages_count_in(topology: *mut c_void, memory_domain_index: usize) -> usize;
    fn fu_huge_pages_count(topology: *mut c_void) -> usize;
    fn fu_memory_domain_id_at_index(topology: *mut c_void, memory_domain_index: usize) -> i32;
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

/// Everything the library can do, whether decided when it was compiled or found on this machine.
///
/// One bit per facility, and two accessors ask two questions of the same bit.
/// [`comptime_capabilities`] reports whether the code was _built_: a set
/// [`Capabilities::PLACE_HUGE_PAGES_ON_DOMAIN`] means we compiled the path that asks for them.
/// [`runtime_capabilities`] reports whether the machine _offers_ it now.
///
/// Neither implies the other. A binary that built [`Capabilities::PLACE_MEMORY_ON_DOMAIN`] runs
/// perfectly well on a single-node box, where the runtime accessor never sets that bit; and a
/// machine with four NUMA nodes reports none of them to a build that left the topology out.
///
/// ```
/// use forkunion::{comptime_capabilities, Capabilities};
/// if comptime_capabilities().contains(Capabilities::COLOCATE_POOLS_ON_DOMAIN) {
///     // `spawn_in`, `DomainAllocator`, and friends are real here.
/// }
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Capabilities(pub u32);

impl Capabilities {
    /// Nothing detected, or nothing compiled in.
    pub const NONE: Capabilities = Capabilities(0);

    /// x86 `pause` instruction.
    pub const X86_PAUSE: Capabilities = Capabilities(1 << 0);
    /// x86-64 `tpause` instruction, with `WAITPKG` support.
    pub const X86_TPAUSE: Capabilities = Capabilities(1 << 1);
    /// Arm `yield` instruction.
    pub const ARM64_YIELD: Capabilities = Capabilities(1 << 2);
    /// AArch64 `wfet` instruction, with `FEAT_WFxT` support.
    pub const ARM64_WFET: Capabilities = Capabilities(1 << 3);
    /// RISC-V `pause` instruction.
    pub const RISC5_PAUSE: Capabilities = Capabilities(1 << 4);
    /// RISC-V `WRS.STO` monitored wait, from the `Zawrs` extension.
    pub const RISC5_WRS: Capabilities = Capabilities(1 << 5);

    /// Own the raw OS thread handle instead of a `std::thread`.
    pub const OS_THREADS: Capabilities = Capabilities(1 << 6);
    /// Enumerate this machine's cores, compute domains, and memory domains.
    pub const TOPOLOGY: Capabilities = Capabilities(1 << 7);
    /// Bind a thread to a set of cores, choosing where it runs.
    pub const PLACE_THREADS_BY_AFFINITY: Capabilities = Capabilities(1 << 8);
    /// Steer a thread onto a class of core at creation, choosing where it runs.
    pub const PLACE_THREADS_BY_CORE_CLASS: Capabilities = Capabilities(1 << 9);
    /// Reclass a thread's scheduler to sleep or wake it, choosing when it runs.
    pub const RESCHEDULE_THREADS_BY_CLASS: Capabilities = Capabilities(1 << 10);
    /// Place a buffer's pages on a chosen memory domain.
    pub const PLACE_MEMORY_ON_DOMAIN: Capabilities = Capabilities(1 << 11);
    /// Place larger-than-base pages on a chosen memory domain.
    pub const PLACE_HUGE_PAGES_ON_DOMAIN: Capabilities = Capabilities(1 << 12);
    /// The kernel promotes base pages to huge pages on its own.
    pub const HUGE_TRANSPARENT_PAGES: Capabilities = Capabilities(1 << 13);
    /// The domain-aware `colocated_pool` and `distributed_pool` are compiled in.
    pub const COLOCATE_POOLS_ON_DOMAIN: Capabilities = Capabilities(1 << 14);

    /// All-ones allow-mask: pass to a pool constructor to disable capability filtering.
    pub const ALL: Capabilities = Capabilities(u32::MAX);

    /// Whether every bit of `other` is set in `self`.
    pub const fn contains(self, other: Capabilities) -> bool {
        (self.0 & other.0) == other.0
    }
}

/// Returns the major version number of the ForkUnion library.
pub fn version_major() -> usize {
    unsafe { fu_version_major() as usize }
}

/// Returns the minor version number of the ForkUnion library.
pub fn version_minor() -> usize {
    unsafe { fu_version_minor() as usize }
}

/// Returns the patch version number of the ForkUnion library.
pub fn version_patch() -> usize {
    unsafe { fu_version_patch() as usize }
}

/// Returns the library version as a tuple of (major, minor, patch).
pub fn version() -> (usize, usize, usize) {
    (version_major(), version_minor(), version_patch())
}

/// Which kernel facilities this build of ForkUnion was compiled to use.
pub fn comptime_capabilities() -> Capabilities {
    Capabilities(unsafe { fu_comptime_capabilities() })
}

/// Formats a capability bitset into a comma-separated name list, like `"threads,topology"`.
///
/// POLISH: returns an owned `String` because `fu_name_capabilities` writes into a caller
/// buffer rather than handing back a static pointer; a fixed-capacity stack type would avoid
/// the allocation. Returns `None` if the C side reports it could not format the names.
#[cfg(feature = "std")]
pub fn name_capabilities(caps: Capabilities) -> Option<std::string::String> {
    let mut buf = [0u8; 256];
    let written =
        unsafe { fu_name_capabilities(caps.0, buf.as_mut_ptr() as *mut c_char, buf.len()) };
    if written == 0 {
        return None;
    }
    let len = core::cmp::min(written, buf.len());
    // Trim any trailing NUL the C side may have written.
    let bytes = &buf[..len];
    let bytes = match bytes.iter().position(|&b| b == 0) {
        Some(nul) => &bytes[..nul],
        None => bytes,
    };
    core::str::from_utf8(bytes)
        .ok()
        .map(std::string::String::from)
}

/// The set [`comptime_capabilities`] bits, comma-separated, like `"threads,topology"`.
#[cfg(feature = "std")]
pub fn comptime_capabilities_string() -> Option<std::string::String> {
    name_capabilities(comptime_capabilities())
}

/// Which features this machine turned out to offer, probing the CPU and the memory system.
pub fn runtime_capabilities() -> Capabilities {
    Capabilities(unsafe { fu_runtime_capabilities() })
}

/// The set [`runtime_capabilities`] bits, comma-separated, like `"arm64_yield,numa_aware"`.
#[cfg(feature = "std")]
pub fn runtime_capabilities_string() -> Option<std::string::String> {
    name_capabilities(runtime_capabilities())
}

/// A position in the topology's array of **compute** domains, in `[0, compute_domains_count())`.
///
/// Distinct from [`MemoryDomain`], and deliberately not interchangeable with it. The two axes are
/// indexed independently: an Apple M5 Pro reports three compute domains over a single memory domain,
/// so a compute index of `2` names no memory domain at all.
///
/// This is not hypothetical. Handing a compute-domain index to `DomainAllocator::new`, which expects a
/// memory-domain index, compiles and works on every machine where the two counts happen to match, then
/// misplaces memory where they diverge. Unlike a C++ `enum`, a newtype also refuses
/// `slice[compute_domain]`, because `Index<usize>` will not accept it - which is the other half of the
/// same bug.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComputeDomain(pub usize);

/// A position in the topology's array of **memory** domains, in `[0, memory_domains_count())`.
/// See [`ComputeDomain`] for why these are separate types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryDomain(pub usize);

/// An OS memory-domain id - a NUMA node - the allocators key off, obtained from
/// [`Topology::memory_domain_id_at_index`]; `-1` when there is none.
///
/// Distinct from [`MemoryDomain`]: that is a dense index for iteration, this is the sparse OS id the
/// kernel labels a node with. A [`DomainAllocator`] holds only this id, so it - and every
/// [`AllocationResult`] it hands out - is free of the topology handle and can outlive it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MemoryDomainId(pub i32);

impl ComputeDomain {
    /// The raw index, for the FFI boundary and for arithmetic.
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }
}

impl MemoryDomain {
    /// The raw index, for the FFI boundary and for arithmetic.
    #[inline]
    pub fn get(self) -> usize {
        self.0
    }
}

impl MemoryDomainId {
    /// The raw OS id, for the FFI boundary; `-1` names no domain.
    #[inline]
    pub fn get(self) -> i32 {
        self.0
    }

    /// Whether this id names a real domain rather than the `-1` sentinel.
    #[inline]
    pub fn is_valid(self) -> bool {
        self.0 >= 0
    }
}

/// An explicit, owned handle to the machine's compute and memory topology.
///
/// The topology is discovered once, when [`Topology::new`] is called, and every affinity query,
/// NUMA allocation, and pool spawn is answered against this handle rather than a process-wide
/// singleton. Create one near the start of a program and share it (`&Topology`) with the pools
/// and allocators that need it; it is [`Send`] + [`Sync`], so a single handle serves every thread.
///
/// # Examples
///
/// ```rust
/// use forkunion::*;
///
/// let topology = Topology::new().expect("Failed to probe topology");
/// let cores = topology.logical_cores_count();
/// let pool = ThreadPool::try_spawn(&topology, cores.max(1)).expect("Failed to spawn pool");
/// assert_eq!(pool.threads_count(), cores.max(1));
/// ```
pub struct Topology {
    inner: *mut c_void,
}

unsafe impl Send for Topology {}
unsafe impl Sync for Topology {}

impl Topology {
    /// Probes the machine and builds a fresh topology handle.
    ///
    /// # Errors
    ///
    /// Returns [`Error::CreationFailed`] if the C side could not allocate or populate the handle.
    pub fn new() -> Result<Self, Error> {
        let inner = unsafe { fu_topology_new() };
        if inner.is_null() {
            return Err(Error::CreationFailed);
        }
        Ok(Self { inner })
    }

    /// Raw C handle, for FFI that takes the topology pointer directly - the symmetric allocator.
    pub(crate) fn raw(&self) -> *mut c_void {
        self.inner
    }

    /// Returns the number of logical cores backing a given compute domain.
    ///
    /// Zero if `compute_domain` is out of range. Use it to size a per-compute-domain pool
    /// ([`ThreadPool::try_spawn_on`]) or to weight work across uneven compute domains.
    pub fn logical_cores_count_in(&self, compute_domain: ComputeDomain) -> usize {
        unsafe { fu_logical_cores_count_in(self.inner, compute_domain.get()) }
    }

    /// Returns the number of logical CPU cores available on the system.
    pub fn logical_cores_count(&self) -> usize {
        unsafe { fu_logical_cores_count(self.inner) }
    }

    /// Returns the number of distinct thread compute_domains available.
    ///
    /// A "compute_domain" represents a group of threads that share the same:
    /// - **NUMA memory domain** - threads with fast local memory access
    /// - **Quality-of-Service level** - P-cores vs E-cores on heterogeneous CPUs
    /// - **Cache hierarchy** - threads sharing L3 cache
    ///
    /// # Typical Values
    ///
    /// - `1` on most desktop, laptop, or IoT platforms with unified memory
    /// - `2-8` on typical dual-socket servers or heterogeneous mobile chips
    /// - `4-32` on high-end cloud servers with multiple sockets
    pub fn compute_domains_count(&self) -> usize {
        unsafe { fu_compute_domains_count(self.inner) }
    }

    /// Returns the performance level of a compute domain (higher = more performant).
    pub fn compute_level_in(&self, compute_domain: ComputeDomain) -> usize {
        unsafe { fu_compute_level_in(self.inner, compute_domain.get()) }
    }

    /// Returns the number of distinct Quality-of-Service levels.
    ///
    /// May be smaller than [`compute_domains_count`](Self::compute_domains_count), as several
    /// domains can share one level - equally-fast cores may still be split across cache clusters,
    /// or across NUMA nodes.
    pub fn compute_levels_count(&self) -> usize {
        unsafe { fu_compute_levels_count(self.inner) }
    }

    /// Returns the relative throughput of one core in a compute domain (0 if unknown).
    ///
    /// A magnitude on the Linux `cpu_capacity` scale, where 1024 is the fastest core present.
    /// This is the number to weight work by - [`compute_level_in`](Self::compute_level_in) is a
    /// dense ordinal and must never be divided by. Platforms that rank cores without rating them
    /// report 0 here; fall back to [`threads_count_in`](ThreadPool::threads_count_in) when they do.
    pub fn compute_capacity_in(&self, compute_domain: ComputeDomain) -> usize {
        unsafe { fu_compute_capacity_in(self.inner, compute_domain.get()) }
    }

    /// Returns the bytes of deepest cache private to a compute domain's cores (0 if unknown).
    ///
    /// Sizes a cache-resident chunk, which is a different question from how many chunks a domain
    /// deserves - domains of equal throughput may back onto very differently sized caches.
    pub fn compute_cache_bytes_in(&self, compute_domain: ComputeDomain) -> usize {
        unsafe { fu_compute_cache_bytes_in(self.inner, compute_domain.get()) }
    }

    /// Returns the number of memory domains available.
    pub fn memory_domains_count(&self) -> usize {
        unsafe { fu_memory_domains_count(self.inner) }
    }

    /// Resolves a memory domain's dense index to the OS id a [`DomainAllocator`] takes.
    ///
    /// Returns [`MemoryDomainId(-1)`](MemoryDomainId) if the index is out of range. The id is the one
    /// place the topology is consulted for allocation; once resolved, the allocator needs it alone.
    pub fn memory_domain_id_at_index(&self, memory_domain: MemoryDomain) -> MemoryDomainId {
        MemoryDomainId(unsafe { fu_memory_domain_id_at_index(self.inner, memory_domain.get()) })
    }

    /// Returns the memory domain nearest a given compute domain (its local allocation target).
    ///
    /// Performance - tiers, latencies, bandwidths, distances - is not the topology's to declare:
    /// harvest a [`Fabric`](crate::Fabric) to measure it in-process.
    pub fn local_memory_of(&self, compute_domain: ComputeDomain) -> MemoryDomain {
        MemoryDomain(unsafe { fu_local_memory_of(self.inner, compute_domain.get()) })
    }

    /// Returns the RAM volume (bytes) held by a given memory domain (0 if out of range).
    pub fn volume_ram_in(&self, memory_domain: MemoryDomain) -> usize {
        unsafe { fu_volume_ram_in(self.inner, memory_domain.get()) }
    }

    /// Returns the total RAM volume (bytes) across all memory domains, regardless of page size.
    pub fn volume_ram(&self) -> usize {
        unsafe { fu_volume_ram(self.inner) }
    }

    /// Returns the huge-page volume (bytes) available on a given memory domain (0 if out of range).
    pub fn volume_huge_pages_in(&self, memory_domain: MemoryDomain) -> usize {
        unsafe { fu_volume_huge_pages_in(self.inner, memory_domain.get()) }
    }

    /// Returns the total huge-page volume (bytes) across all memory domains.
    pub fn volume_huge_pages(&self) -> usize {
        unsafe { fu_volume_huge_pages(self.inner) }
    }

    /// Returns the number of free huge pages in a given memory domain (0 if out of range).
    pub fn huge_pages_count_in(&self, memory_domain: MemoryDomain) -> usize {
        unsafe { fu_huge_pages_count_in(self.inner, memory_domain.get()) }
    }

    /// Returns the total number of free huge pages across all memory domains.
    pub fn huge_pages_count(&self) -> usize {
        unsafe { fu_huge_pages_count(self.inner) }
    }
}

impl Drop for Topology {
    fn drop(&mut self) {
        unsafe {
            fu_topology_delete(self.inner);
        }
    }
}

/// Defines whether the calling thread participates in task execution.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CallerExclusivity {
    /// The calling thread participates in the workload (spawns N-1 workers)
    Inclusive = 0,
    /// The calling thread only coordinates, doesn't execute tasks (spawns N workers)
    Exclusive = 1,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::*;

    #[inline]
    pub(crate) fn hw_threads() -> usize {
        let topology = Topology::new().unwrap();
        topology.logical_cores_count().max(1)
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn capabilities() {
        let topology = Topology::new().unwrap();
        let comptime = comptime_capabilities();
        let runtime = runtime_capabilities();
        std::println!("Comptime: {:?}", comptime_capabilities_string());
        std::println!("Runtime:  {:?}", runtime_capabilities_string());
        assert!(runtime_capabilities_string().is_some());

        // Threads are the one facility every supported platform has.
        assert!(comptime.contains(Capabilities::OS_THREADS));

        // The aggregate is implied, never hand-set: pools need threads and a topology to spawn onto.
        assert_eq!(
            comptime.contains(Capabilities::COLOCATE_POOLS_ON_DOMAIN),
            comptime.contains(Capabilities::OS_THREADS)
                && comptime.contains(Capabilities::TOPOLOGY)
        );

        // Placing pages on a node presumes we discovered the nodes.
        if comptime.contains(Capabilities::PLACE_MEMORY_ON_DOMAIN) {
            assert!(comptime.contains(Capabilities::TOPOLOGY));
        }

        // Whatever we can construct, we can only construct because a capability was compiled in.
        if !comptime.contains(Capabilities::COLOCATE_POOLS_ON_DOMAIN) {
            assert_eq!(topology.compute_domains_count(), 1);
        }

        // One facility, two questions of the same bit: a machine can only _offer_ page placement if
        // this build compiled the path that asks for it.
        if runtime.contains(Capabilities::PLACE_MEMORY_ON_DOMAIN) {
            assert!(comptime.contains(Capabilities::PLACE_MEMORY_ON_DOMAIN));
        }
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn system_info() {
        let topology = Topology::new().unwrap();
        let cores = topology.logical_cores_count();
        let numa = topology.memory_domains_count();
        let compute_domains = topology.compute_domains_count();
        let qos = topology.compute_levels_count();

        std::println!(
            "Cores: {cores}, NUMA: {numa}, ComputeDomains: {compute_domains}, QoS: {qos}"
        );
        assert!(cores > 0);
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn topology_axes() {
        let topology = Topology::new().unwrap();
        let compute_domains = topology.compute_domains_count();
        let compute_levels = topology.compute_levels_count();
        let memory_domains = topology.memory_domains_count();
        assert!(compute_domains > 0 && memory_domains > 0);

        // Levels are dense ranks over domains, so they can never outnumber them.
        assert!(compute_levels <= compute_domains);

        for domain in (0..compute_domains).map(ComputeDomain) {
            assert!(topology.compute_level_in(domain) < compute_levels.max(1));
            assert!(topology.local_memory_of(domain).get() < memory_domains);
            // Capacity and cache are magnitudes, unknown as 0 - never negative, never asserted nonzero.
            let _capacity = topology.compute_capacity_in(domain);
            let _cache_bytes = topology.compute_cache_bytes_in(domain);
        }

        // Out-of-range indices must saturate to 0 rather than trap or read past the topology.
        assert_eq!(
            topology.compute_capacity_in(ComputeDomain(compute_domains + 64)),
            0
        );
        assert_eq!(
            topology.compute_cache_bytes_in(ComputeDomain(compute_domains + 64)),
            0
        );
    }
}
