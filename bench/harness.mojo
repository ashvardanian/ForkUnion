"""The settings both Mojo benchmarks share, mirroring `bench/harness.hpp`.

Both benchmarks read every variable once at start, and a value that does not parse prints one line
and exits with status 1. Leaving a variable unset or empty keeps its default.

    Variable                           Default                  Meaning
    FORKUNION_SEED                     42                       Seed of every draw, or random
    FORKUNION_WARMUP                   1s                       Untimed run before timing
    FORKUNION_TIME_LIMIT               10s                      Timed window, like 500ms
    FORKUNION_THREADS                  all cores                Threads; 0 also means all cores
    FORKUNION_BACKEND                  forkunion_static_shared  Backend, per benchmark header
    FORKUNION_NBODY_COUNT              the thread count         Bodies to simulate
    FORKUNION_PROPAGATION_SCALE        14                       2^scale vertices per community
    FORKUNION_PROPAGATION_COMMUNITIES  64                       Communities strung on the ring
    FORKUNION_PROPAGATION_EDGE_FACTOR  16                       Raw edges per vertex
    FORKUNION_PROPAGATION_CHECK        false                    Also converge serially to check
"""

from std.os import getenv
from std.random import random_ui64, seed as reseed
from std.sys import exit, stderr
from std.time import perf_counter_ns


def env_text(name: StaticString) -> Optional[String]:
    """The text of the environment variable `name`, or nothing when it is unset or empty."""
    var text = getenv(name)
    if text.byte_length() == 0:
        return None
    return text


def exit_unparsed(name: StaticString, text: String, expected: StaticString):
    """Prints `NAME="text" does not parse, expected <expected>` and exits with status 1."""
    print(t"{name}=\"{text}\" does not parse, expected {expected}", file=stderr)
    exit(1)


def parse_count(text: String) -> Optional[Int]:
    """A positive whole number in ASCII digits, like "64"."""
    var bytes = text.as_bytes()
    if len(bytes) == 0 or len(bytes) > 18:
        return None
    var count = 0
    for index in range(len(bytes)):
        var digit = Int(bytes[index]) - 48
        if digit < 0 or digit > 9:
            return None
        count = count * 10 + digit
    if count == 0:
        return None
    return count


def parse_duration(text: String) -> Optional[Int]:
    """A positive duration in whole milliseconds or seconds, like "200ms" or "10s", in ns."""
    var in_ms = text.endswith("ms")
    if not in_ms and not text.endswith("s"):
        return None
    var count = parse_count(String(text.removesuffix("ms" if in_ms else "s")))
    if not count or count.value() > 0xFFFF_FFFF:
        return None
    return count.value() * (1_000_000 if in_ms else 1_000_000_000)


def parse_seed(text: String) -> Optional[UInt32]:
    """A 32-bit seed, or "random" for a fresh draw seeded from the clock."""
    if text == "random":
        reseed()
        return UInt32(random_ui64(0, 0xFFFF_FFFF))
    if text == "0":
        return UInt32(0)
    var count = parse_count(text)
    if not count or count.value() > 0xFFFF_FFFF:
        return None
    return UInt32(count.value())


def env_count(name: StaticString, fallback: Int) -> Int:
    var text = env_text(name)
    if not text:
        return fallback
    var count = parse_count(text.value())
    if not count:
        exit_unparsed(name, text.value(), "a positive count")
    return count.value()


def env_duration(name: StaticString, fallback_ns: Int) -> Int:
    var text = env_text(name)
    if not text:
        return fallback_ns
    var duration = parse_duration(text.value())
    if not duration:
        exit_unparsed(name, text.value(), "a duration like 200ms or 10s")
    return duration.value()


def env_flag(name: StaticString, fallback: Bool) -> Bool:
    var text = env_text(name)
    if not text:
        return fallback
    if text.value() == "1" or text.value() == "true":
        return True
    if text.value() != "0" and text.value() != "false":
        exit_unparsed(name, text.value(), "0, 1, true or false")
    return False


def env_seed(name: StaticString, fallback: UInt32) -> UInt32:
    var text = env_text(name)
    if not text:
        return fallback
    var seed = parse_seed(text.value())
    if not seed:
        exit_unparsed(name, text.value(), "an unsigned integer or random")
    return seed.value()


def env_threads(name: StaticString, all_cores: Int) -> Int:
    """Reads a thread count like "8", or `all_cores` when it is unset, empty or "0"."""
    var text = env_text(name)
    if not text or text.value() == "0":
        return all_cores
    var count = parse_count(text.value())
    if not count:
        exit_unparsed(name, text.value(), "a count, 0 for all cores")
    return count.value()


def spell_duration(ns: Int) -> String:
    """Spells `ns` as a user types it: whole seconds as "10s", anything else as "1500ms"."""
    var ms = ns // 1_000_000
    if ms % 1000 == 0:
        return String(ms // 1000) + "s"
    return String(ms) + "ms"


def fixed(value: Float64, decimals: Int) -> String:
    """Renders a non-negative value with exactly `decimals` places, since t-strings take no spec."""
    var scale = 1
    for _ in range(decimals):
        scale *= 10
    var scaled = Int(value * Float64(scale) + 0.5)
    var whole = String(scaled // scale)
    var fraction = String(scaled % scale)
    while fraction.byte_length() < decimals:
        fraction = String("0") + fraction
    return whole + "." + fraction


def spell_rate(label: String, per_second: Float64) -> String:
    """Spells `per_second` of `label` in decimal units, like "  edges 2.622 M/s"."""
    var scaled = per_second
    var prefix = String("")
    for next in ["k", "M", "G", "T", "P"]:
        if scaled >= 1000:
            scaled /= 1000
            prefix = String(next)
    return "  " + label + " " + fixed(scaled, 3) + " " + prefix + "/s"


@fieldwise_init
struct Counter(Copyable):
    """A named amount per call: printed per second when `is_rate`, or as it is."""

    var name: String
    var value: Float64
    var is_rate: Bool


@fieldwise_init
struct Row(Movable):
    """A finished benchmark: its name, calls per second and counters."""

    var name: String
    var calls_per_second: Float64
    var counters: List[Counter]


def print_row(row: Row):
    """Prints `row` on one line, with rates in decimal units like "M/s"."""
    var line = row.name
    while line.byte_length() < 56:
        line += " "
    line += spell_rate("calls", row.calls_per_second)
    for counter in row.counters:
        if counter.is_rate:
            line += spell_rate(counter.name, counter.value * row.calls_per_second)
        else:
            line += "  " + counter.name + " " + fixed(counter.value, 3)
    print(line)


struct Loop(Movable):
    """One benchmark's timed loop, iterated as `while loop.next():`.

    Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
    until the time limit. It reads the clock once per 64th of the calls so far, so a short call
    doesn't time the clock itself.
    """

    var warmup_ns: Int
    var time_limit_ns: Int
    var start_ns: Int
    var elapsed_ns: Int
    var calls: Int
    var next_check: Int
    var warming_up: Bool
    var counters: List[Counter]

    def __init__(out self, warmup_ns: Int, time_limit_ns: Int):
        self.warmup_ns = warmup_ns
        self.time_limit_ns = time_limit_ns
        self.start_ns = Int(perf_counter_ns())
        self.elapsed_ns = 0
        self.calls = 0
        self.next_check = 0
        self.warming_up = True
        self.counters = List[Counter]()

    def next(mut self) -> Bool:
        """Whether to run one more call, False once the time limit has passed."""
        if self.calls >= self.next_check:
            self.elapsed_ns = Int(perf_counter_ns()) - self.start_ns
            if self.warming_up and self.elapsed_ns >= self.warmup_ns:
                self.warming_up = False
                self.start_ns = Int(perf_counter_ns())
                self.elapsed_ns = 0
                self.calls = 0
            elif not self.warming_up and self.elapsed_ns >= self.time_limit_ns:
                return False
            self.next_check = self.calls + self.calls // 64 + 1
        self.calls += 1
        return True

    def rate(mut self, name: String, per_call: Float64):
        """Reports `per_call` of `name` per second."""
        self.counters.append(Counter(name, per_call, True))

    def counter(mut self, name: String, value: Float64):
        """Reports `value` as `name`, unscaled."""
        self.counters.append(Counter(name, value, False))

    def row(self, name: String) -> Row:
        """The finished benchmark under `name`."""
        return Row(name, Float64(self.calls) / (Float64(self.elapsed_ns) / 1e9), self.counters.copy())


struct Settings:
    """Every benchmark setting, read once from the environment by the constructor."""

    var seed: UInt32
    var warmup_ns: Int
    var time_limit_ns: Int
    var threads: Int
    var backend: String
    var bodies: Int
    var scale: Int
    var communities: Int
    var edge_factor: Int
    var check: Bool

    def __init__(out self, all_cores: Int):
        """Reads every `FORKUNION_*` variable, exiting with status 1 on the first unparsable one."""
        self.seed = env_seed("FORKUNION_SEED", 42)
        self.warmup_ns = env_duration("FORKUNION_WARMUP", 1_000_000_000)
        self.time_limit_ns = env_duration("FORKUNION_TIME_LIMIT", 10_000_000_000)
        self.threads = env_threads("FORKUNION_THREADS", all_cores)
        self.backend = env_text("FORKUNION_BACKEND").or_else(String("forkunion_static_shared"))
        self.bodies = env_count("FORKUNION_NBODY_COUNT", self.threads)
        self.scale = env_count("FORKUNION_PROPAGATION_SCALE", 14)
        self.communities = env_count("FORKUNION_PROPAGATION_COMMUNITIES", 64)
        self.edge_factor = env_count("FORKUNION_PROPAGATION_EDGE_FACTOR", 16)
        self.check = env_flag("FORKUNION_PROPAGATION_CHECK", False)


def print_settings(settings: Settings):
    """Prints each setting as "- Name: value", in the grammar it parses from."""
    var warmup = spell_duration(settings.warmup_ns)
    var time_limit = spell_duration(settings.time_limit_ns)
    var check = "true" if settings.check else "false"
    print(t"- Seed: {settings.seed}")
    print(t"- Warm-up: {warmup}")
    print(t"- Time limit: {time_limit}")
    print(t"- Threads: {settings.threads}")
    print(t"- Backend: {settings.backend}")
    print(t"- Bodies: {settings.bodies}")
    print(t"- Scale: {settings.scale}")
    print(t"- Communities: {settings.communities}")
    print(t"- Edge factor: {settings.edge_factor}")
    print(t"- Check: {check}")
