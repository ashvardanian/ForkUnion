fn main() -> Result<(), cc::Error> {
    let mut build = cc::Build::new();

    // Detect features and target for NUMA enablement
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let numa_feature_enabled = std::env::var("CARGO_FEATURE_NUMA").is_ok();
    let enable_numa = target_os == "linux" && numa_feature_enabled;

    build
        .cpp(true) // Enable C++ support
        .std("c++17") // Use C++17 standard
        .file("c/forkunion.cpp")
        .include("include")
        .define("FU_ENABLE_NUMA", if enable_numa { "1" } else { "0" })
        .opt_level(2) // Optimize compiled C++ to -O2
        .flag_if_supported("-pedantic") // Only for GCC/Clang
        .warnings(false);

    // Compile the C++ library first, so Cargo emits
    // `-lstatic=forkunion` before we add dependent libs.
    if let Err(e) = build.try_compile("forkunion") {
        print!("cargo:warning={e}");
        return Err(e);
    }

    // Important: add dependent system libraries AFTER the static lib.
    // For GNU ld, static libraries are resolved left-to-right, so
    // `-lnuma -lpthread` must appear after `-lforkunion` to satisfy symbols.
    if enable_numa {
        // Link against `libnuma` when NUMA is enabled on Linux
        println!("cargo:rustc-link-lib=numa");
    }

    // Always link `pthreads` on Linux since the library uses std::thread internally
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=pthread");
    }

    println!("cargo:rerun-if-changed=c/forkunion.cpp");
    println!("cargo:rerun-if-changed=rust/forkunion.rs");
    println!("cargo:rerun-if-changed=include/forkunion.h");
    println!("cargo:rerun-if-changed=include/forkunion.hpp");
    Ok(())
}
