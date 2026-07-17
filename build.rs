//! Compiles the C++ core and decides which kernel facilities it may use.
//!
//! The derivation rules live in `include/forkunion/types.hpp`, not here. By default this script
//! defines no `FU_WITH_*` macro at all and lets the header work it out from the platform. Cargo
//! features only ever *override* that.
//!
//! Nothing here probes the build host. The core reads its topology from sysfs and places memory by
//! syscall, so there is no `libnuma` to find and no artifact that differs by where it was built.
//!
//! Features are additive, which is an awkward fit for a switch that wants three positions, so:
//!
//! - default: AUTO. The header decides, and this script says nothing.
//! - `topology`, `place-memory-on-domain`, `place-huge-pages-on-domain`, `place-threads-by-affinity`:
//!   force the capability on. A build that cannot honour it - or a prerequisite it needs, like huge
//!   pages needing on-domain placement - stops at the `#error` in `types.hpp`, with a sentence,
//!   rather than at link time with missing symbols.
//! - `portable`: force every optional capability off, leaving the STL thread pool. Useful under
//!   musl, inside containers, and for seeing what a caller on an unsupported platform will see.

use std::path::Path;

/// Every optional capability, in the order `types.hpp` declares them.
const OPTIONAL_CAPABILITIES: [&str; 6] = [
    "FU_WITH_TOPOLOGY",
    "FU_WITH_PLACE_THREADS_BY_AFFINITY",
    "FU_WITH_PLACE_THREADS_BY_CORE_CLASS",
    "FU_WITH_RESCHEDULE_THREADS_BY_CLASS",
    "FU_WITH_PLACE_MEMORY_ON_DOMAIN",
    "FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN",
];

fn main() -> Result<(), cc::Error> {
    let mut build = cc::Build::new();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let portable = std::env::var("CARGO_FEATURE_PORTABLE").is_ok();
    let force_topology = std::env::var("CARGO_FEATURE_TOPOLOGY").is_ok();
    let force_place_memory_on_domain =
        std::env::var("CARGO_FEATURE_PLACE_MEMORY_ON_DOMAIN").is_ok();
    let force_place_huge_pages_on_domain =
        std::env::var("CARGO_FEATURE_PLACE_HUGE_PAGES_ON_DOMAIN").is_ok();
    let force_place_threads_by_affinity =
        std::env::var("CARGO_FEATURE_PLACE_THREADS_BY_AFFINITY").is_ok();

    build
        .cpp(true) // Enable C++ support
        .std("c++17") // Use C++17 standard
        .file("c/forkunion.cpp")
        .include("include")
        .flag_if_supported("-pedantic") // Only for GCC/Clang
        .warnings(false);

    if portable {
        assert!(
            !force_topology
                && !force_place_memory_on_domain
                && !force_place_huge_pages_on_domain
                && !force_place_threads_by_affinity,
            "`portable` turns off the very capabilities the other features turn on"
        );
        for capability in OPTIONAL_CAPABILITIES {
            build.define(capability, "0");
        }
    } else {
        if force_topology {
            build.define("FU_WITH_TOPOLOGY", "1");
        }
        if force_place_memory_on_domain {
            build.define("FU_WITH_PLACE_MEMORY_ON_DOMAIN", "1");
        }
        if force_place_huge_pages_on_domain {
            build.define("FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN", "1");
        }
        if force_place_threads_by_affinity {
            build.define("FU_WITH_PLACE_THREADS_BY_AFFINITY", "1");
        }
    }

    // Mirror Rust's `debug_assertions` onto the C++ `assert`s. Cargo only exports this as a
    // `cfg`, and it tracks the profile's `debug-assertions` key - unlike `DEBUG`, which is
    // debug-info and stays `true` under `[profile.release] debug = true`.
    if std::env::var_os("CARGO_CFG_DEBUG_ASSERTIONS").is_none() {
        build.define("NDEBUG", None);
    }

    // Compile the C++ library first, so Cargo emits
    // `-lstatic=forkunion` before we add dependent libs.
    if let Err(e) = build.try_compile("forkunion") {
        print!("cargo:warning={e}");
        return Err(e);
    }

    // Important: add dependent system libraries AFTER the static lib.
    // For GNU ld, static libraries are resolved left-to-right, so
    // `-lpthread` must appear after `-lforkunion` to satisfy symbols.
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=pthread");
    }

    // Hand dependents the headers, so a crate compiling its own C against the `fu_*` ABI need not
    // vendor a copy that drifts; `links = "forkunion"` makes this their `DEP_FORKUNION_INCLUDE`.
    // Anchored to the manifest, not the cwd, so it resolves inside a published crate too.
    println!(
        "cargo:include={}",
        Path::new(
            &std::env::var("CARGO_MANIFEST_DIR").expect("Cargo always sets CARGO_MANIFEST_DIR")
        )
        .join("include")
        .display()
    );

    println!("cargo:rerun-if-changed=c/forkunion.cpp");
    println!("cargo:rerun-if-changed=rust/forkunion.rs");
    println!("cargo:rerun-if-changed=include/forkunion.h");
    println!("cargo:rerun-if-changed=include/forkunion.hpp");
    // The umbrella only `#include`s; the implementation lives in the sub-headers beside it.
    println!("cargo:rerun-if-changed=include/forkunion");
    Ok(())
}
