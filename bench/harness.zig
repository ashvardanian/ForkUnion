//! The settings both Zig benchmarks share, mirroring `bench/harness.hpp`.
//!
//! Both benchmarks read every variable once at start, and a value that does not parse prints one
//! line and exits with status 1. Leaving a variable unset or empty keeps its default.
//!
//!     Variable                           Default                  Meaning
//!     FORKUNION_SEED                     42                       Seed of every draw, or random
//!     FORKUNION_WARMUP                   1s                       Untimed run before timing
//!     FORKUNION_TIME_LIMIT               10s                      Timed window, like 500ms
//!     FORKUNION_THREADS                  all cores                Threads; 0 also means all cores
//!     FORKUNION_BACKEND                  forkunion_static_shared  Backend, per benchmark header
//!     FORKUNION_NBODY_COUNT              the thread count         Bodies to simulate
//!     FORKUNION_PROPAGATION_SCALE        14                       2^scale vertices per community
//!     FORKUNION_PROPAGATION_COMMUNITIES  64                       Communities strung on the ring
//!     FORKUNION_PROPAGATION_EDGE_FACTOR  16                       Raw edges per vertex
//!     FORKUNION_PROPAGATION_CHECK        false                    Also converge serially to check

const std = @import("std");

extern "c" fn getentropy(buffer: *anyopaque, length: usize) c_int;

/// The text of the environment variable `name`, or null when it is unset or empty; borrows libc's
/// storage.
pub fn envText(name: [*:0]const u8) ?[]const u8 {
    const text = std.mem.span(std.c.getenv(name) orelse return null);
    return if (text.len == 0) null else text;
}

/// Prints `NAME="text" does not parse, expected <expected>` and exits with status 1.
pub fn exitUnparsed(name: [*:0]const u8, text: []const u8, expected: []const u8) noreturn {
    std.debug.print("{s}=\"{s}\" does not parse, expected {s}\n", .{ name, text, expected });
    std.process.exit(1);
}

/// Parses `name` with `parse`, or returns `fallback` when it is unset or empty; exits if it does
/// not parse.
pub fn envParsed(comptime T: type, name: [*:0]const u8, fallback: T, comptime parse: fn ([]const u8) ?T, expected: []const u8) T {
    const text = envText(name) orelse return fallback;
    return parse(text) orelse exitUnparsed(name, text, expected);
}

/// A positive whole number in ASCII digits, like "64".
pub fn parseCount(text: []const u8) ?usize {
    if (text.len == 0) return null;
    for (text) |byte| if (!std.ascii.isDigit(byte)) return null;
    const count = std.fmt.parseUnsigned(usize, text, 10) catch return null;
    return if (count == 0) null else count;
}

/// A positive duration in whole milliseconds or seconds, like "200ms" or "10s", as nanoseconds.
pub fn parseDuration(text: []const u8) ?u64 {
    const in_ms = std.mem.endsWith(u8, text, "ms");
    if (!in_ms and !std.mem.endsWith(u8, text, "s")) return null;
    const count = parseCount(text[0 .. text.len - @as(usize, if (in_ms) 2 else 1)]) orelse return null;
    return std.math.mul(u64, count, if (in_ms) std.time.ns_per_ms else std.time.ns_per_s) catch null;
}

/// A 32-bit seed, or "random" for 32 bits from the OS entropy source.
pub fn parseSeed(text: []const u8) ?u32 {
    if (std.mem.eql(u8, text, "random")) {
        var seed: u32 = 0;
        return if (getentropy(&seed, @sizeOf(u32)) == 0) seed else null;
    }
    for (text) |byte| if (!std.ascii.isDigit(byte)) return null;
    return std.fmt.parseUnsigned(u32, text, 10) catch null;
}

fn parseFlag(text: []const u8) ?bool {
    if (std.mem.eql(u8, text, "1") or std.mem.eql(u8, text, "true")) return true;
    if (std.mem.eql(u8, text, "0") or std.mem.eql(u8, text, "false")) return false;
    return null;
}

pub fn envCount(name: [*:0]const u8, fallback: usize) usize {
    return envParsed(usize, name, fallback, parseCount, "a positive count");
}

pub fn envDuration(name: [*:0]const u8, fallback_ns: u64) u64 {
    return envParsed(u64, name, fallback_ns, parseDuration, "a duration like 200ms or 10s");
}

pub fn envFlag(name: [*:0]const u8, fallback: bool) bool {
    return envParsed(bool, name, fallback, parseFlag, "0, 1, true or false");
}

pub fn envSeed(name: [*:0]const u8, fallback: u32) u32 {
    return envParsed(u32, name, fallback, parseSeed, "an unsigned integer or random");
}

/// Reads a thread count like "8", or `all_cores` when it is unset, empty or "0".
pub fn envThreads(name: [*:0]const u8, all_cores: usize) usize {
    const text = envText(name) orelse return all_cores;
    if (std.mem.eql(u8, text, "0")) return all_cores;
    return parseCount(text) orelse exitUnparsed(name, text, "a count, 0 for all cores");
}

/// Spells `ns` as a user types it: whole seconds as "10s", anything else as "1500ms".
pub fn spellDuration(buffer: []u8, ns: u64) []const u8 {
    const ms = ns / std.time.ns_per_ms;
    const spelled = if (ms % 1000 == 0) std.fmt.bufPrint(buffer, "{}s", .{ms / 1000}) else std.fmt.bufPrint(buffer, "{}ms", .{ms});
    return spelled catch buffer[0..0];
}

/// Writes one preformatted line to STDOUT; diagnostics stay on STDERR via `std.debug.print`.
pub fn writeStdout(text: []const u8) void {
    _ = std.c.write(std.Io.File.stdout().handle, text.ptr, text.len);
}

fn monotonicNanos() u64 {
    var timespec: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &timespec);
    return @as(u64, @intCast(timespec.sec)) * std.time.ns_per_s + @as(u64, @intCast(timespec.nsec));
}

/// A named amount per call: printed per second when `is_rate`, or as it is.
pub const Counter = struct {
    name: []const u8 = "",
    value: f64 = 0,
    is_rate: bool = false,
};

/// A finished benchmark: its name, calls per second and counters.
pub const Row = struct {
    name: []const u8,
    calls_per_second: f64,
    counters: [4]Counter,

    /// Prints the row on one line, with rates in decimal units like "M/s".
    pub fn print(self: Row) void {
        var buffer: [512]u8 = undefined;
        var line: std.Io.Writer = .fixed(&buffer);
        line.print("{s:<56}", .{self.name}) catch return;
        printRate(&line, "calls", self.calls_per_second);
        for (self.counters) |entry| {
            if (entry.name.len == 0) continue;
            if (entry.is_rate) {
                printRate(&line, entry.name, entry.value * self.calls_per_second);
            } else {
                line.print("  {s} {d:.3}", .{ entry.name, entry.value }) catch return;
            }
        }
        line.writeByte('\n') catch return;
        writeStdout(line.buffered());
    }

    fn printRate(line: *std.Io.Writer, label: []const u8, per_second: f64) void {
        var scaled = per_second;
        var prefix: []const u8 = "";
        for ([_][]const u8{ "k", "M", "G", "T", "P" }) |next| {
            if (scaled >= 1000) {
                scaled /= 1000;
                prefix = next;
            }
        }
        line.print("  {s} {d:.3} {s}/s", .{ label, scaled, prefix }) catch {};
    }
};

/// One benchmark's timed loop, iterated as `while (loop.next()) |call|`.
///
/// Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
/// until the time limit. It reads the clock once per 64th of the calls so far, so a short call
/// doesn't time the clock itself.
pub const Loop = struct {
    warmup_ns: u64,
    time_limit_ns: u64,
    start_ns: u64,
    elapsed_ns: u64 = 0,
    calls: usize = 0,
    next_check: usize = 0,
    warming_up: bool = true,
    counters: [4]Counter = .{Counter{}} ** 4,

    pub fn init(warmup_ns: u64, time_limit_ns: u64) Loop {
        return .{ .warmup_ns = warmup_ns, .time_limit_ns = time_limit_ns, .start_ns = monotonicNanos() };
    }

    /// The index of the next call, or null once the time limit has passed.
    pub fn next(self: *Loop) ?usize {
        if (self.calls >= self.next_check) {
            self.elapsed_ns = monotonicNanos() - self.start_ns;
            if (self.warming_up and self.elapsed_ns >= self.warmup_ns) {
                self.warming_up = false;
                self.start_ns = monotonicNanos();
                self.elapsed_ns = 0;
                self.calls = 0;
            } else if (!self.warming_up and self.elapsed_ns >= self.time_limit_ns) return null;
            self.next_check = self.calls + self.calls / 64 + 1;
        }
        self.calls += 1;
        return self.calls - 1;
    }

    fn add(self: *Loop, entry: Counter) void {
        for (&self.counters) |*slot| {
            if (slot.name.len != 0) continue;
            slot.* = entry;
            return;
        }
    }

    /// Reports `per_call` of `name` per second.
    pub fn rate(self: *Loop, name: []const u8, per_call: f64) void {
        self.add(.{ .name = name, .value = per_call, .is_rate = true });
    }

    /// Reports `value` as `name`, unscaled.
    pub fn counter(self: *Loop, name: []const u8, value: f64) void {
        self.add(.{ .name = name, .value = value });
    }

    /// The finished benchmark under `name`.
    pub fn row(self: Loop, name: []const u8) Row {
        const seconds = @as(f64, @floatFromInt(self.elapsed_ns)) / std.time.ns_per_s;
        return .{ .name = name, .calls_per_second = @as(f64, @floatFromInt(self.calls)) / seconds, .counters = self.counters };
    }
};

/// Every benchmark setting, its default as the initializer, filled once by `read`.
pub const Settings = struct {
    seed: u32 = 42,
    warmup_ns: u64 = 1 * std.time.ns_per_s,
    time_limit_ns: u64 = 10 * std.time.ns_per_s,
    threads: usize,
    backend: []const u8 = "forkunion_static_shared",
    bodies: usize,
    scale: usize = 14,
    communities: usize = 64,
    edge_factor: usize = 16,
    check: bool = false,

    /// Reads every `FORKUNION_*` variable, exiting with status 1 on the first that does not parse.
    pub fn read(all_cores: usize) Settings {
        var settings: Settings = .{ .threads = all_cores, .bodies = all_cores };
        settings.seed = envSeed("FORKUNION_SEED", settings.seed);
        settings.warmup_ns = envDuration("FORKUNION_WARMUP", settings.warmup_ns);
        settings.time_limit_ns = envDuration("FORKUNION_TIME_LIMIT", settings.time_limit_ns);
        settings.threads = envThreads("FORKUNION_THREADS", settings.threads);
        settings.backend = envText("FORKUNION_BACKEND") orelse settings.backend;
        settings.bodies = envCount("FORKUNION_NBODY_COUNT", settings.threads);
        settings.scale = envCount("FORKUNION_PROPAGATION_SCALE", settings.scale);
        settings.communities = envCount("FORKUNION_PROPAGATION_COMMUNITIES", settings.communities);
        settings.edge_factor = envCount("FORKUNION_PROPAGATION_EDGE_FACTOR", settings.edge_factor);
        settings.check = envFlag("FORKUNION_PROPAGATION_CHECK", settings.check);
        return settings;
    }

    /// Prints each setting as "- Name: value", in the grammar it parses from.
    pub fn print(self: Settings) void {
        var warmup_buffer: [32]u8 = undefined;
        var time_limit_buffer: [32]u8 = undefined;
        var line_buffer: [512]u8 = undefined;
        const lines = std.fmt.bufPrint(&line_buffer,
            \\- Seed: {}
            \\- Warm-up: {s}
            \\- Time limit: {s}
            \\- Threads: {}
            \\- Backend: {s}
            \\- Bodies: {}
            \\- Scale: {}
            \\- Communities: {}
            \\- Edge factor: {}
            \\- Check: {s}
            \\
        , .{
            self.seed,
            spellDuration(&warmup_buffer, self.warmup_ns),
            spellDuration(&time_limit_buffer, self.time_limit_ns),
            self.threads,
            self.backend,
            self.bodies,
            self.scale,
            self.communities,
            self.edge_factor,
            if (self.check) "true" else "false",
        }) catch return;
        writeStdout(lines);
    }
};
