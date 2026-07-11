const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Check Zig version compatibility (requires 0.16.0 or later)
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @panic("ForkUnion requires Zig 0.16.0 or later. Please upgrade your Zig toolchain.");
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Which kernel facilities the C++ core may use.
    //
    // The derivation rules live in `include/forkunion/types.hpp`, not here. Left alone, each option
    // is `null` and we pass no `-DFU_WITH_*` at all, so the header decides from the platform and
    // from whether `<numa.h>` is there to include. `-Dnuma-memory=true` and friends only override
    // that; an override the platform cannot honour stops at an `#error`, not at link time.
    const with_topology = b.option(bool, "topology", "Enumerate compute and memory domains");
    const with_place_memory_on_domain = b.option(bool, "place-memory-on-domain", "Place pages on a chosen memory domain");
    const with_place_huge_pages_on_domain = b.option(bool, "place-huge-pages-on-domain", "Request pages larger than the base page");
    const with_place_threads_by_affinity = b.option(bool, "place-threads-by-affinity", "Bind worker threads to cores");
    const portable = b.option(bool, "portable", "Force every optional capability off") orelse false;

    // Compile the C++ library from c/forkunion.cpp (like Rust's build.rs does)
    const lib = b.addLibrary(.{
        .name = "forkunion",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });

    // Build C++ flags
    var cpp_flags: std.ArrayList([]const u8) = .empty;
    defer cpp_flags.deinit(b.allocator);
    cpp_flags.appendSlice(b.allocator, &.{ "-std=c++20", "-fno-exceptions", "-fno-rtti" }) catch @panic("OOM");

    const optional_capabilities = [_][]const u8{
        "FU_WITH_TOPOLOGY",                    "FU_WITH_PLACE_THREADS_BY_AFFINITY",
        "FU_WITH_PLACE_THREADS_BY_CORE_CLASS", "FU_WITH_RESCHEDULE_THREADS_BY_CLASS",
        "FU_WITH_PLACE_MEMORY_ON_DOMAIN",      "FU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN",
    };

    const numa_memory = with_place_memory_on_domain;
    if (portable) {
        if (with_topology == true or numa_memory == true or with_place_huge_pages_on_domain == true or with_place_threads_by_affinity == true)
            @panic("-Dportable turns off the very capabilities the other options turn on");
        for (optional_capabilities) |capability|
            cpp_flags.append(b.allocator, b.fmt("-D{s}=0", .{capability})) catch @panic("OOM");
    } else {
        if (with_topology) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_TOPOLOGY={d}", .{@intFromBool(on)})) catch @panic("OOM");
        if (numa_memory) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_PLACE_MEMORY_ON_DOMAIN={d}", .{@intFromBool(on)})) catch @panic("OOM");
        if (with_place_huge_pages_on_domain) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_PLACE_HUGE_PAGES_ON_DOMAIN={d}", .{@intFromBool(on)})) catch @panic("OOM");
        if (with_place_threads_by_affinity) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_PLACE_THREADS_BY_AFFINITY={d}", .{@intFromBool(on)})) catch @panic("OOM");
    }

    // We link `libnuma` whenever the target could want it, and let the header decide whether to call
    // it. Over-linking costs a `DT_NEEDED` entry that `--as-needed` drops; under-linking costs a
    // wall of undefined symbols the caller cannot trace back to a missing package.
    const link_numa = target.result.os.tag == .linux and !portable and numa_memory != false;

    lib.root_module.addCSourceFile(.{
        .file = b.path("c/forkunion.cpp"),
        .flags = cpp_flags.items,
    });

    lib.root_module.addIncludePath(b.path("include"));
    lib.root_module.link_libcpp = true; // Use Zig's bundled `libc++` instead of system `libstdc++`

    b.installArtifact(lib);

    // Create forkunion module for use as a dependency
    _ = b.addModule("forkunion", .{
        .root_source_file = b.path("zig/forkunion.zig"),
        .target = target,
    });

    // Unit tests
    const test_step = b.step("test", "Run library tests");
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/forkunion.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    lib_tests.root_module.addIncludePath(b.path("include"));
    lib_tests.root_module.linkLibrary(lib);
    if (target.result.os.tag == .linux) {
        lib_tests.root_module.linkSystemLibrary("pthread", .{});
        if (link_numa) {
            lib_tests.root_module.linkSystemLibrary("numa", .{});
        }
    }

    const run_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_tests.step);
}
