const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const enable_numa = b.option(bool, "numa", "Enable NUMA support (Linux only)") orelse
        (target.result.os.tag == .linux);

    // Get the forkunion module and artifact from parent
    const forkunion_dep = b.dependency("forkunion", .{
        .target = target,
        .optimize = optimize,
    });
    const forkunion_module = forkunion_dep.module("forkunion");
    const forkunion_artifact = forkunion_dep.artifact("forkunion");

    // N-body benchmark executable
    const nbody = b.addExecutable(.{
        .name = "forkunion_nbody",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nbody.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    nbody.root_module.link_libc = true;
    nbody.root_module.link_libcpp = true;
    nbody.root_module.linkLibrary(forkunion_artifact);
    if (target.result.os.tag == .linux) {
        nbody.root_module.linkSystemLibrary("pthread", .{});
        if (enable_numa) {
            nbody.root_module.linkSystemLibrary("numa", .{});
        }
    }
    nbody.root_module.addImport("forkunion", forkunion_module);

    // Add benchmark dependencies
    if (b.lazyDependency("libxev", .{
        .target = target,
        .optimize = optimize,
    })) |libxev_dep| {
        nbody.root_module.addImport("xev", libxev_dep.module("xev"));
    }

    b.installArtifact(nbody);

    const run_nbody = b.addRunArtifact(nbody);
    if (b.args) |args| {
        run_nbody.addArgs(args);
    }

    const run_step = b.step("run", "Run N-body benchmark");
    run_step.dependOn(&run_nbody.step);

    // The Connected-Components benchmark by label propagation - same graph, kernel, and output
    // contract as `propagation.cpp` and `propagation.rs`, bit-identical across all three.
    const propagation = b.addExecutable(.{
        .name = "forkunion_propagation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("propagation.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    propagation.root_module.link_libc = true;
    propagation.root_module.link_libcpp = true;
    propagation.root_module.linkLibrary(forkunion_artifact);
    if (target.result.os.tag == .linux) {
        propagation.root_module.linkSystemLibrary("pthread", .{});
        if (enable_numa) {
            propagation.root_module.linkSystemLibrary("numa", .{});
        }
    }
    propagation.root_module.addImport("forkunion", forkunion_module);

    b.installArtifact(propagation);

    const run_propagation = b.addRunArtifact(propagation);
    if (b.args) |args| {
        run_propagation.addArgs(args);
    }

    const run_propagation_step = b.step("run-propagation", "Run the label-propagation benchmark");
    run_propagation_step.dependOn(&run_propagation.step);
}
