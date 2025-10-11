const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Get the fork_union module and artifact from parent
    const fork_union_dep = b.dependency("fork_union", .{
        .target = target,
        .optimize = optimize,
    });
    const fork_union_module = fork_union_dep.module("fork_union");
    const fork_union_artifact = fork_union_dep.artifact("fork_union");

    // N-body benchmark executable
    const nbody = b.addExecutable(.{
        .name = "nbody_zig",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nbody.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    nbody.linkLibC();
    nbody.linkLibCpp();
    nbody.linkLibrary(fork_union_artifact);
    nbody.root_module.addImport("fork_union", fork_union_module);

    // Add optional benchmark dependencies
    if (b.lazyDependency("spice", .{
        .target = target,
        .optimize = optimize,
    })) |spice_dep| {
        nbody.root_module.addImport("spice", spice_dep.module("spice"));
    }

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
}
