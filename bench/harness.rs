//! The settings both Rust benchmarks share, mirroring `bench/harness.hpp`.
//!
//! Both benchmarks read every variable once at start, and a value that does not parse prints one
//! line and exits with status 1. Leaving a variable unset or empty keeps its default.
//!
//! ```text
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
//! ```
//!
//! File: bench/harness.rs
//! Author: Ash Vardanian
use std::env;
use std::time::{Duration, Instant};

/// Reads `name`, or `None` when it is unset or empty.
pub fn env_text(name: &str) -> Option<String> {
    env::var(name).ok().filter(|text| !text.is_empty())
}

/// Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does
/// not parse.
pub fn env_parsed<T>(name: &str, fallback: T, parse: impl FnOnce(&str) -> Option<T>, expected: &str) -> T {
    let Some(text) = env_text(name) else {
        return fallback;
    };
    parse(&text).unwrap_or_else(|| {
        eprintln!("{name}=\"{text}\" does not parse, expected {expected}");
        std::process::exit(1)
    })
}

/// Reads a positive count like `128`, or `fallback` when unset or empty; exits if it does not
/// parse.
pub fn env_count(name: &str, fallback: usize) -> usize {
    env_parsed(name, fallback, parse_count, "a positive count")
}

/// Reads a duration like `200ms` or `10s`, or `fallback` when unset or empty; exits if it does not
/// parse.
pub fn env_duration(name: &str, fallback: Duration) -> Duration {
    env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s")
}

/// Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty; exits if it does not
/// parse.
pub fn env_flag(name: &str, fallback: bool) -> bool {
    let parse = |text: &str| match text {
        "0" | "false" => Some(false),
        "1" | "true" => Some(true),
        _ => None,
    };
    env_parsed(name, fallback, parse, "0, 1, true or false")
}

/// Reads a 32-bit seed or `random`, or `fallback` when unset or empty; exits if it does not parse.
pub fn env_seed(name: &str, fallback: u32) -> u32 {
    env_parsed(name, fallback, parse_seed, "an unsigned integer or random")
}

/// Parses a 32-bit unsigned integer, or `random` as 32 bits from the OS entropy source.
pub fn parse_seed(text: &str) -> Option<u32> {
    if text == "random" {
        use std::hash::{BuildHasher, Hasher};

        return Some(std::hash::RandomState::new().build_hasher().finish() as u32);
    }
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten()
}

/// Parses a positive whole number in ASCII digits, like `128`; zero is `None`.
pub fn parse_count(text: &str) -> Option<usize> {
    let digits = !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit());
    digits.then(|| text.parse().ok()).flatten().filter(|&count| count != 0)
}

/// Parses a duration like `200ms` or `10s`; a bare number, a fraction or zero is `None`.
pub fn parse_duration(text: &str) -> Option<Duration> {
    match text.strip_suffix("ms") {
        Some(count) => parse_count(count).map(|count| Duration::from_millis(count as u64)),
        None => parse_count(text.strip_suffix('s')?).map(|count| Duration::from_secs(count as u64)),
    }
}

/// Spells a duration the way `parse_duration` reads it: `1s`, `1500ms`.
pub fn spell_duration(duration: Duration) -> String {
    let milliseconds = duration.as_millis();
    match milliseconds % 1000 {
        0 => format!("{}s", milliseconds / 1000),
        _ => format!("{milliseconds}ms"),
    }
}

/// A named amount per call: printed per second when `is_rate`, or as it is.
#[derive(Clone, Copy, Default)]
pub struct Counter {
    pub name: &'static str,
    pub value: f64,
    pub is_rate: bool,
}

/// A finished benchmark: its name, calls per second and counters.
pub struct Row<'a> {
    pub name: &'a str,
    pub calls_per_second: f64,
    pub counters: [Counter; 4],
}

impl Row<'_> {
    /// Prints the row on one line, with rates in decimal units like "M/s".
    pub fn print(&self) {
        let print_rate = |label: &str, mut per_second: f64| {
            let mut prefix = "";
            for next in ["k", "M", "G", "T", "P"] {
                if per_second >= 1000.0 {
                    per_second /= 1000.0;
                    prefix = next;
                }
            }
            print!("  {label} {per_second:.3} {prefix}/s");
        };
        print!("{:<56}", self.name);
        print_rate("calls", self.calls_per_second);
        for counter in self.counters.iter().filter(|counter| !counter.name.is_empty()) {
            match counter.is_rate {
                true => print_rate(counter.name, counter.value * self.calls_per_second),
                false => print!("  {} {:.3}", counter.name, counter.value),
            }
        }
        println!();
    }
}

/// One benchmark's timed loop, iterated as `for call in &mut timed`.
///
/// Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
/// until the time limit. It reads the clock once per 64th of the calls so far, so a short call
/// doesn't time the clock itself.
pub struct Loop {
    warmup: Duration,
    time_limit: Duration,
    start: Instant,
    elapsed: Duration,
    calls: usize,
    next_check: usize,
    warming_up: bool,
    counters: [Counter; 4],
}

impl Loop {
    pub fn new(warmup: Duration, time_limit: Duration) -> Self {
        Loop {
            warmup,
            time_limit,
            start: Instant::now(),
            elapsed: Duration::ZERO,
            calls: 0,
            next_check: 0,
            warming_up: true,
            counters: [Counter::default(); 4],
        }
    }

    fn add(&mut self, counter: Counter) {
        if let Some(slot) = self.counters.iter_mut().find(|slot| slot.name.is_empty()) {
            *slot = counter;
        }
    }

    /// Reports `per_call` of `name` per second.
    pub fn rate(&mut self, name: &'static str, per_call: f64) {
        self.add(Counter {
            name,
            value: per_call,
            is_rate: true,
        });
    }

    /// Reports `value` as `name`, unscaled.
    #[allow(dead_code)]
    pub fn counter(&mut self, name: &'static str, value: f64) {
        self.add(Counter {
            name,
            value,
            is_rate: false,
        });
    }

    /// The finished benchmark under `name`.
    pub fn row<'a>(&self, name: &'a str) -> Row<'a> {
        Row {
            name,
            calls_per_second: self.calls as f64 / self.elapsed.as_secs_f64(),
            counters: self.counters,
        }
    }
}

impl Iterator for Loop {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.calls >= self.next_check {
            self.elapsed = self.start.elapsed();
            if self.warming_up && self.elapsed >= self.warmup {
                self.warming_up = false;
                self.start = Instant::now();
                self.elapsed = Duration::ZERO;
                self.calls = 0;
            } else if !self.warming_up && self.elapsed >= self.time_limit {
                return None;
            }
            self.next_check = self.calls + self.calls / 64 + 1;
        }
        self.calls += 1;
        Some(self.calls - 1)
    }
}

/// Every benchmark setting, read once by `Settings::read`.
pub struct Settings {
    pub seed: u32,
    pub warmup: Duration,
    pub time_limit: Duration,
    pub threads: usize,
    pub backend: String,
    pub bodies: usize,
    pub scale: usize,
    pub communities: usize,
    pub edge_factor: usize,
    pub check: bool,
}

impl Settings {
    /// Reads every `FORKUNION_*` variable, exiting with status 1 on the first that does not parse.
    /// Unset, empty or `0` threads resolve to `all_cores`, the topology's logical core count.
    pub fn read(all_cores: usize) -> Self {
        let parse_threads = |text: &str| match text {
            "0" => Some(all_cores),
            _ => parse_count(text),
        };
        let threads = env_parsed(
            "FORKUNION_THREADS",
            all_cores,
            parse_threads,
            "a count, 0 for all cores",
        );
        Settings {
            seed: env_seed("FORKUNION_SEED", 42),
            warmup: env_duration("FORKUNION_WARMUP", Duration::from_secs(1)),
            time_limit: env_duration("FORKUNION_TIME_LIMIT", Duration::from_secs(10)),
            threads,
            backend: env_text("FORKUNION_BACKEND").unwrap_or_else(|| "forkunion_static_shared".into()),
            bodies: env_count("FORKUNION_NBODY_COUNT", threads),
            scale: env_count("FORKUNION_PROPAGATION_SCALE", 14),
            communities: env_count("FORKUNION_PROPAGATION_COMMUNITIES", 64),
            edge_factor: env_count("FORKUNION_PROPAGATION_EDGE_FACTOR", 16),
            check: env_flag("FORKUNION_PROPAGATION_CHECK", false),
        }
    }

    /// Prints each setting as "- Name: value", in the grammar it parses from.
    pub fn print(&self) {
        println!("- Seed: {}", self.seed);
        println!("- Warm-up: {}", spell_duration(self.warmup));
        println!("- Time limit: {}", spell_duration(self.time_limit));
        println!("- Threads: {}", self.threads);
        println!("- Backend: {}", self.backend);
        println!("- Bodies: {}", self.bodies);
        println!("- Scale: {}", self.scale);
        println!("- Communities: {}", self.communities);
        println!("- Edge factor: {}", self.edge_factor);
        println!("- Check: {}", self.check);
    }
}
