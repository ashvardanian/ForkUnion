const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    // Check Zig version compatibility (requires 0.16.0 or later)
    if (builtin.zig_version.major == 0 and builtin.zig_version.minor < 16) {
        @panic("Fork Union requires Zig 0.16.0 or later. Please upgrade your Zig toolchain.");
    }

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Determine NUMA support
    const enable_numa = b.option(bool, "numa", "Enable NUMA support (Linux only)") orelse
        (target.result.os.tag == .linux);

    // Compile the C++ library from c/lib.cpp (like Rust's build.rs does)
    const lib = b.addLibrary(.{
        .name = "fork_union",
        .linkage = .static,
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });

    // Build C++ flags
    const cpp_flags = if (enable_numa and target.result.os.tag == .linux)
        &[_][]const u8{
            "-std=c++20",
            "-fno-exceptions",
            "-fno-rtti",
            "-DFU_ENABLE_NUMA=1",
        }
    else
        &[_][]const u8{
            "-std=c++20",
            "-fno-exceptions",
            "-fno-rtti",
            "-DFU_ENABLE_NUMA=0",
        };

    lib.addCSourceFile(.{
        .file = b.path("c/lib.cpp"),
        .flags = cpp_flags,
    });

    lib.addIncludePath(b.path("include"));
    lib.linkLibCpp(); // Use Zig's bundled `libc++` instead of system `libstdc++`

    b.installArtifact(lib);

    // Create fork_union module for use as a dependency
    _ = b.addModule("fork_union", .{
        .root_source_file = b.path("zig/fork_union.zig"),
        .target = target,
    });

    // Unit tests
    const test_step = b.step("test", "Run library tests");
    const lib_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/fork_union.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    lib_tests.addIncludePath(b.path("include"));
    lib_tests.linkLibrary(lib);
    if (target.result.os.tag == .linux) {
        lib_tests.root_module.linkSystemLibrary("pthread", .{});
        if (enable_numa) {
            lib_tests.root_module.linkSystemLibrary("numa", .{});
        }
    }

    const run_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_tests.step);
}
