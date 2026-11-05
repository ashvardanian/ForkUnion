//! Compiles the C++ core through CMake, the one place that probes which capabilities the compiler
//! builds, and decides which kernel facilities it may use.
//!
//! The derivation rules live in `include/forkunion/types.hpp`, not here. By default this script
//! forces no `FORKUNION_WITH_*` option at all and lets the header work it out from the platform.
//! Cargo features only ever _override_ that.
//!
//! Nothing here probes the build host. The core reads its topology from sysfs and places memory by
//! syscall, so there is no `libnuma` to find and no artifact that differs by where it was built.
//!
//! Features are additive, which is an awkward fit for a switch that wants three positions, so:
//!
//! - `default`: AUTO. The header decides, and this script says nothing.
//! - `topology`, `place-memory-on-domain`, `place-huge-pages-on-domain`,
//!   `place-threads-by-affinity`: force the capability on. A build that cannot honour it - or a
//!   prerequisite it needs, like huge pages needing on-domain placement - stops at the `#error` in
//!   `types.hpp`, with a sentence, rather than at link time with missing symbols.
//! - `portable`: force every optional capability off, leaving the STL thread pool. Useful under
//!   musl, inside containers, and for seeing what a caller on an unsupported platform will see.
//!
//! File: build.rs
//! Author: Ash Vardanian
//! Date: June 29, 2025

use std::{env, path::PathBuf};

/// Each Cargo feature and the CMake cache string it forces on.
const FEATURE_OPTIONS: [(&str, &str); 4] = [
    ("topology", "FORKUNION_WITH_TOPOLOGY"),
    ("place-memory-on-domain", "FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN"),
    (
        "place-huge-pages-on-domain",
        "FORKUNION_WITH_PLACE_HUGE_PAGES_ON_DOMAIN",
    ),
    ("place-threads-by-affinity", "FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY"),
];

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("Cargo always sets CARGO_MANIFEST_DIR"));
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let enabled = |feature: &str| {
        let variable = format!("CARGO_FEATURE_{}", feature.to_uppercase().replace('-', "_"));
        env::var_os(variable).is_some()
    };

    let mut library = cmake::Config::new(&manifest);
    library
        .profile("Release")
        .define("FORKUNION_INSTALL", "ON")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("FORKUNION_PORTABLE", if enabled("portable") { "ON" } else { "OFF" });
    for (feature, option) in FEATURE_OPTIONS {
        if enabled(feature) {
            library.define(option, "ON");
        }
    }
    // The default `noeh` libc++abi has no `__cxa_throw`, so exceptions stay off.
    if target_os == "wasi" {
        library.cxxflag("-fno-exceptions");
    }
    let prefix = library.build();
    println!("cargo:rustc-link-search=native={}", prefix.join("lib").display());
    println!("cargo:rustc-link-lib=static=forkunion_static");

    // The archive is C++, so the standard library it was compiled against has to come along.
    match target_os.as_str() {
        "windows" => {}
        "macos" | "ios" | "freebsd" => println!("cargo:rustc-link-lib=dylib=c++"),
        "android" => println!("cargo:rustc-link-lib=dylib=c++_shared"),
        // wasi-sdk ships libc++ and libc++abi as archives under its sysroot, in the `noeh`
        // multilib.
        "wasi" => {
            println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");
            let target = env::var("TARGET").expect("Cargo always sets TARGET");
            let sdk = PathBuf::from(env::var("WASI_SDK_PATH").expect("WASI builds need WASI_SDK_PATH"));
            let sysroot = sdk.join("share/wasi-sysroot/lib").join(target).join("noeh");
            println!("cargo:rustc-link-search=native={}", sysroot.display());
            println!("cargo:rustc-link-lib=static=c++");
            println!("cargo:rustc-link-lib=static=c++abi");
        }
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }

    // Hand dependents the headers, so a crate compiling its own C against the `fu_*` ABI need not
    // vendor a copy that drifts; `links = "forkunion"` makes this their `DEP_FORKUNION_INCLUDE`.
    // Anchored to the manifest, not the cwd, so it resolves inside a published crate too. The
    // header roots are joined like `PATH`, for a dependent to split with `env::split_paths`.
    let roots = env::join_paths(["c", "include"].map(|root| manifest.join(root)))
        .expect("the manifest path holds no `PATH` separator");
    println!("cargo:include={}", roots.to_string_lossy());
    // The whole tree, so a dependent's CMake adds the checkout Cargo resolved as a subdirectory.
    println!("cargo:source={}", manifest.display());

    for source in [
        "CMakeLists.txt",
        "VERSION",
        "cmake",
        "probes",
        "c",
        "include",
        "src",
        "rust/forkunion.rs",
    ] {
        println!("cargo:rerun-if-changed={}", manifest.join(source).display());
    }
}
