const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Check Zig version compatibility (requires 0.15.0 or later)
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 15) {
        @panic("ForkUnion requires Zig 0.15.0 or later. Please upgrade your Zig toolchain.");
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Which kernel facilities the C++ core may use.
    //
    // The derivation rules live in `include/forkunion/types.hpp`, not here. Left alone, each option
    // is `null` and we pass no `-DFU_WITH_*` at all, so the header decides from the platform and
    // from whether `<numa.h>` is there to include. `-Dnuma-memory=true` and friends only override
    // that; an override the platform cannot honour stops at an `#error`, not at link time.
    const with_numa_memory = b.option(bool, "numa-memory", "Place pages on a chosen memory domain");
    const with_huge_pages = b.option(bool, "huge-pages", "Request pages larger than the base page");
    const with_topology = b.option(bool, "topology", "Enumerate compute and memory domains");
    const portable = b.option(bool, "portable", "Force every optional capability off") orelse false;

    // Deprecated: `-Dnuma` used to mean five things at once. Forward it to the one it mostly meant.
    const legacy_numa = b.option(bool, "numa", "Deprecated: use -Dnuma-memory");

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
        "FU_WITH_TOPOLOGY",   "FU_WITH_TOPOLOGY_CACHES",    "FU_WITH_TOPOLOGY_METRICS", "FU_WITH_THREAD_PINNING",
        "FU_WITH_THREAD_QOS", "FU_WITH_THREAD_SCHED_CLASS", "FU_WITH_NUMA_MEMORY",      "FU_WITH_HUGE_PAGES",
    };

    const numa_memory = with_numa_memory orelse legacy_numa;
    if (portable) {
        if (numa_memory == true or with_huge_pages == true or with_topology == true)
            @panic("-Dportable turns off the very capabilities the other options turn on");
        for (optional_capabilities) |capability|
            cpp_flags.append(b.allocator, b.fmt("-D{s}=0", .{capability})) catch @panic("OOM");
    } else {
        if (with_topology) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_TOPOLOGY={d}", .{@intFromBool(on)})) catch @panic("OOM");
        if (numa_memory) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_NUMA_MEMORY={d}", .{@intFromBool(on)})) catch @panic("OOM");
        if (with_huge_pages) |on| cpp_flags.append(b.allocator, b.fmt("-DFU_WITH_HUGE_PAGES={d}", .{@intFromBool(on)})) catch @panic("OOM");
    }

    // We link `libnuma` whenever the target could want it, and let the header decide whether to call
    // it. Over-linking costs a `DT_NEEDED` entry that `--as-needed` drops; under-linking costs a
    // wall of undefined symbols the caller cannot trace back to a missing package.
    const link_numa = target.result.os.tag == .linux and !portable and numa_memory != false;

    lib.addCSourceFile(.{
        .file = b.path("c/forkunion.cpp"),
        .flags = cpp_flags.items,
    });

    lib.addIncludePath(b.path("include"));
    lib.linkLibCpp(); // Use Zig's bundled `libc++` instead of system `libstdc++`

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

    lib_tests.addIncludePath(b.path("include"));
    lib_tests.linkLibrary(lib);
    if (target.result.os.tag == .linux) {
        lib_tests.root_module.linkSystemLibrary("pthread", .{});
        if (link_numa) {
            lib_tests.root_module.linkSystemLibrary("numa", .{});
        }
    }

    const run_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_tests.step);
}
