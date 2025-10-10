const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Determine NUMA support
    const enable_numa = b.option(bool, "numa", "Enable NUMA support (Linux only)") orelse
        (target.result.os.tag == .linux);

    // Compile the C++ library from c/lib.cpp (like Rust's build.rs does)
    const lib = b.addStaticLibrary(.{
        .name = "fork_union",
        .target = target,
        .optimize = optimize,
    });

    lib.addCSourceFile(.{
        .file = b.path("c/lib.cpp"),
        .flags = &.{
            "-std=c++20",
            "-fno-exceptions",
            "-fno-rtti",
        },
    });

    lib.addIncludePath(b.path("include"));
    lib.linkLibCpp(); // Use Zig's bundled `libc++` instead of system `libstdc++`

    if (enable_numa and target.result.os.tag == .linux) {
        lib.defineCMacro("FU_ENABLE_NUMA", "1");
        lib.linkSystemLibrary("numa");
        lib.linkSystemLibrary("pthread");
    } else {
        lib.defineCMacro("FU_ENABLE_NUMA", "0");
        if (target.result.os.tag == .linux) {
            lib.linkSystemLibrary("pthread");
        }
    }

    b.installArtifact(lib);

    // Create fork_union module that links against our compiled library
    const fork_union_module = b.addModule("fork_union", .{
        .root_source_file = b.path("zig/fork_union.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Unit tests
    const test_step = b.step("test", "Run library tests");
    const lib_tests = b.addTest(.{
        .root_source_file = b.path("zig/fork_union.zig"),
        .target = target,
        .optimize = optimize,
    });

    lib_tests.addIncludePath(b.path("include"));
    lib_tests.linkLibrary(lib);

    const run_tests = b.addRunArtifact(lib_tests);
    test_step.dependOn(&run_tests.step);

    // N-body benchmark example
    const nbody = b.addExecutable(.{
        .name = "nbody_zig",
        .root_source_file = b.path("scripts/nbody.zig"),
        .target = target,
        .optimize = optimize,
    });

    nbody.linkLibC();
    nbody.root_module.addImport("fork_union", fork_union_module);
    nbody.addIncludePath(b.path("include"));
    nbody.linkLibrary(lib);
    nbody.linkSystemLibrary("stdc++");

    const nbody_install = b.addInstallArtifact(nbody, .{});
    const nbody_step = b.step("nbody", "Build N-body benchmark");
    nbody_step.dependOn(&nbody_install.step);

    const run_nbody = b.addRunArtifact(nbody);
    if (b.args) |args| {
        run_nbody.addArgs(args);
    }

    const run_nbody_step = b.step("run-nbody", "Run N-body benchmark");
    run_nbody_step.dependOn(&run_nbody.step);
}
