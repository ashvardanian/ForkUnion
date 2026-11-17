/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief The settings, machine report and timed loop the C++ benchmarks share.
 *
 *  Both benchmarks read every setting once at start, from a @c --name flag or else its
 *  @c FORKUNION_NAME variable, and a value that does not parse prints one line and exits with
 *  status 1. Leaving one unset or empty keeps its default. The settings are listed under
 *  "Benchmarks" in CONTRIBUTING.md.
 */
#pragma once
#include <cctype>  // `std::tolower`
#include <cmath>   // `std::isfinite`
#include <cstdint> // `std::uint64_t`
#include <cstdio>  // `std::fprintf`, `std::printf`
#include <cstdlib> // `std::getenv`, `std::exit`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <regex>        // `std::regex`
#include <string>       // `std::string`, `std::to_string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`

#include "forkunion.hpp" // `FORKUNION_VERSION_*`, `capability_name`, `allowed_cores_count`

namespace fu = ashvardanian::forkunion;

namespace ashvardanian::forkunion::bench {

using steady_clock_t = std::chrono::steady_clock;
using time_point_t = steady_clock_t::time_point;

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    std::fprintf(stderr, "%s=\"%s\" does not parse, expected %s\n", variable, text, expected);
    std::exit(1);
}

/** The text of a setting: its @p flag when given, else its environment @p variable , else null. */
inline char const *setting_text(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    for (int index = 1; index < argc; ++index) {
        std::string_view const argument {argv[index]};
        if (argument == flag)
            return index + 1 < argc && std::strncmp(argv[index + 1], "--", 2) != 0 ? argv[index + 1] : "1";
        if (argument.size() > flag.size() && argument.starts_with(flag) && argument[flag.size()] == '=')
            return argv[index] + flag.size() + 1;
    }
    char const *text = std::getenv(variable);
    return text && *text ? text : nullptr;
}

/** A switch, @p fallback when unset: 0, false, off and no turn it off, anything else on. */
inline bool read_flag_setting(int argc, char **argv, char const *variable, std::string_view flag,
                              bool fallback = false) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    for (char const *off : {"0", "false", "off", "no"})
        if (std::strcmp(text, off) == 0) return false;
    return true;
}

/** A whole number, @p fallback when unset. */
inline std::size_t read_count_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                      std::size_t fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "a whole number");
    return count;
}

/** A positive real, @p fallback when unset. */
inline double read_real_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                double fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    char *end = nullptr;
    double const value = std::strtod(text, &end);
    if (*end != '\0' || !(value > 0) || !std::isfinite(value))
        exit_unparsed(variable, text, "a positive number like 0.1 or 10");
    return value;
}

/** A positive duration such as 200ms, 10s or 2m, @p fallback when unset. */
inline std::chrono::milliseconds read_duration_setting(int argc, char **argv, char const *variable,
                                                       std::string_view flag,
                                                       std::chrono::milliseconds fallback) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return fallback;
    std::string_view const spelled {text};
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (error == std::errc {} && count != 0) {
        if (unit == "ms") return std::chrono::milliseconds(count);
        if (unit == "s") return std::chrono::seconds(count);
        if (unit == "m") return std::chrono::minutes(count);
    }
    exit_unparsed(variable, text, "a duration like 200ms, 10s or 2m");
}

/** The run's seed: 42 when unset, drawn afresh for @c random , else any unsigned 64-bit value. */
inline std::uint64_t read_seed_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return 42;
    std::string_view const spelled {text};
    if (spelled == "random") {
        std::random_device device;
        return (static_cast<std::uint64_t>(device()) << 32) ^ device();
    }
    std::uint64_t seed = 0;
    auto const [end, error] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (error != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: a regex search when the pattern compiles, else a substring. */
struct filter_t {
    std::string pattern;
    std::optional<std::regex> regex;

    bool selects(std::string_view label) const noexcept {
        if (pattern.empty()) return true;
        if (regex) return std::regex_search(label.begin(), label.end(), *regex);
        return label.find(pattern) != std::string_view::npos;
    }
};

/** The filter a setting spells, compiled once; empty, selecting everything, when unset. */
inline filter_t read_filter_setting(int argc, char **argv, char const *variable, std::string_view flag) noexcept {
    filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.pattern = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.emplace(filter.pattern);
    }
    catch (std::regex_error const &) {
    }
#else
    filter.regex.emplace(filter.pattern);
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    filter_t filter;
    char const *program = "";
};

/** SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits. */
constexpr std::uint64_t split_mix_bits(std::uint64_t value) noexcept {
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ull;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBull;
    return value ^ (value >> 31);
}

/** The @p index -th stream of the family @p name under @p seed , identical in every language. */
constexpr std::uint64_t stream_key(std::uint64_t seed, std::string_view name, std::uint64_t index = 0) noexcept {
    std::uint64_t hashed = 0xCBF29CE484222325ull;
    for (char const character : name) hashed = (hashed ^ static_cast<std::uint8_t>(character)) * 0x100000001B3ull;
    return split_mix_bits(split_mix_bits(seed ^ hashed) + index);
}

/**
 *  @brief SplitMix64 over one 64-bit state, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t state = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : state(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(state += 0x9E3779B97F4A7C15ull); }

    /** A uniform draw below @p bound , zero for an empty bound, without the bias of a modulo. */
    std::uint64_t below(std::uint64_t bound) noexcept {
        if (bound == 0) return 0;
        std::uint64_t const rejection_floor = (0u - bound) % bound;
        std::uint64_t draw = (*this)();
        while (draw < rejection_floor) draw = (*this)();
        return draw % bound;
    }
};

#pragma endregion Shared Settings

/** A count past zero, @p fallback when unset, as a zero would leave nothing to time. */
inline std::size_t read_positive_count_setting(int argc, char **argv, char const *variable, std::string_view flag,
                                               std::size_t fallback) noexcept {
    std::size_t const count = read_count_setting(argc, argv, variable, flag, fallback);
    if (count == 0) exit_unparsed(variable, setting_text(argc, argv, variable, flag), "a positive count");
    return count;
}

/** Spells @p duration as a user types it: whole seconds as "10s", anything else as "1500ms". */
inline std::string spell_duration(std::chrono::milliseconds duration) {
    auto const count = duration.count();
    return count % 1000 ? std::to_string(count) + "ms" : std::to_string(count / 1000) + "s";
}

/** A named amount per call: printed per second when @c is_rate, or as it is. */
struct counter_t {
    char const *name = nullptr;
    double value = 0;
    bool is_rate = false;
};

/** A finished benchmark: its name, calls per second and counters. */
struct row_t {
    std::string_view name;
    double calls_per_second = 0;
    std::array<counter_t, 4> counters {};
};

/** Prints @p row on one line, with rates in decimal units like "M/s". */
inline void print(row_t const &row) noexcept {
    auto const print_rate = [](char const *label, double per_second) noexcept {
        char const *prefix = "";
        for (char const *next : {"k", "M", "G", "T", "P"})
            if (per_second >= 1000) per_second /= 1000, prefix = next;
        std::printf("  %s %.3f %s/s", label, per_second, prefix);
    };
    std::printf("%-56.*s", static_cast<int>(row.name.size()), row.name.data());
    print_rate("calls", row.calls_per_second);
    for (counter_t const &counter : row.counters)
        if (counter.is_rate) print_rate(counter.name, counter.value * row.calls_per_second);
        else if (counter.name) std::printf("  %s %.3f", counter.name, counter.value);
    std::printf("\n");
}

/**
 *  @brief One benchmark's timed loop, iterated as `for (std::size_t call : loop)`.
 *
 *  Setup above the loop stays untimed. The loop runs untimed for the warm-up, then counts calls
 *  until the time limit. It reads the clock once per 64th of the calls so far, so a short call
 *  doesn't time the clock itself.
 */
class loop_t {
    std::chrono::milliseconds warmup_, time_limit_;
    time_point_t start_ {};
    steady_clock_t::duration elapsed_ {};
    std::size_t calls_ = 0, next_check_ = 0;
    bool warming_up_ = true;
    std::array<counter_t, 4> counters_ {};

    bool keep_running() noexcept {
        if (calls_ < next_check_) return true;
        elapsed_ = steady_clock_t::now() - start_;
        if (warming_up_ && elapsed_ >= warmup_)
            warming_up_ = false, start_ = steady_clock_t::now(), elapsed_ = {}, calls_ = 0;
        else if (!warming_up_ && elapsed_ >= time_limit_) return false;
        next_check_ = calls_ + calls_ / 64 + 1;
        return true;
    }

    void add(counter_t counter) noexcept {
        for (counter_t &slot : counters_)
            if (!slot.name) return void(slot = counter);
    }

  public:
    struct end_t {};
    struct iterator_t {
        loop_t *loop;
        bool operator!=(end_t) noexcept { return loop->keep_running(); }
        std::size_t operator*() const noexcept { return loop->calls_; }
        void operator++() noexcept { ++loop->calls_; }
    };

    loop_t(std::chrono::milliseconds warmup, std::chrono::milliseconds time_limit) noexcept
        : warmup_(warmup), time_limit_(time_limit) {}

    iterator_t begin() noexcept { return start_ = steady_clock_t::now(), iterator_t {this}; }
    end_t end() const noexcept { return {}; }

    /** Reports @p per_call of @p name per second. */
    void rate(char const *name, double per_call) noexcept { add({name, per_call, true}); }

    /** Reports @p value as @p name, unscaled. */
    void counter(char const *name, double value) noexcept { add({name, value, false}); }

    /** The finished benchmark under @p name. */
    row_t row(std::string_view name) const noexcept {
        return {name, static_cast<double>(calls_) / std::chrono::duration<double>(elapsed_).count(), counters_};
    }
};

/** Every benchmark setting, its default as the initializer, filled by @c read_bench_settings. */
struct bench_settings_t : common_settings_t {
    std::chrono::milliseconds warmup = std::chrono::seconds(1);
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);
    std::size_t threads = allowed_cores_count();
    char const *backend = "forkunion_static_shared";
    std::size_t bodies = threads;
    std::size_t scale = 14;
    std::size_t communities = 64;
    std::size_t edge_factor = 16;
    bool check = false;
};

/** Reads every @c bench_settings_t field; a thread count of 0 means every allowed core. */
inline bench_settings_t read_bench_settings(int argc, char **argv) noexcept {
    bench_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "FORKUNION_SEED", "--seed");
    settings.warmup = read_duration_setting(argc, argv, "FORKUNION_WARMUP", "--warmup", settings.warmup);
    settings.time_limit =
        read_duration_setting(argc, argv, "FORKUNION_TIME_LIMIT", "--time-limit", settings.time_limit);
    settings.threads = read_count_setting(argc, argv, "FORKUNION_THREADS", "--threads", settings.threads);
    if (settings.threads == 0) settings.threads = allowed_cores_count();
    if (char const *backend = setting_text(argc, argv, "FORKUNION_BACKEND", "--backend")) settings.backend = backend;
    settings.bodies =
        read_positive_count_setting(argc, argv, "FORKUNION_NBODY_COUNT", "--nbody-count", settings.threads);
    settings.scale =
        read_positive_count_setting(argc, argv, "FORKUNION_PROPAGATION_SCALE", "--propagation-scale", settings.scale);
    settings.communities = read_positive_count_setting(argc, argv, "FORKUNION_PROPAGATION_COMMUNITIES",
                                                       "--propagation-communities", settings.communities);
    settings.edge_factor = read_positive_count_setting(argc, argv, "FORKUNION_PROPAGATION_EDGE_FACTOR",
                                                       "--propagation-edge-factor", settings.edge_factor);
    settings.check = read_flag_setting(argc, argv, "FORKUNION_PROPAGATION_CHECK", "--propagation-check");
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(bench_settings_t const &settings) {
    std::printf("- Seed: %llu\n", static_cast<unsigned long long>(settings.seed));
    std::printf("- Warm-up: %s\n", spell_duration(settings.warmup).c_str());
    std::printf("- Time limit: %s\n", spell_duration(settings.time_limit).c_str());
    std::printf("- Threads: %zu\n", settings.threads);
    std::printf("- Backend: %s\n", settings.backend);
    std::printf("- Bodies: %zu\n", settings.bodies);
    std::printf("- Scale: %zu\n", settings.scale);
    std::printf("- Communities: %zu\n", settings.communities);
    std::printf("- Edge factor: %zu\n", settings.edge_factor);
    std::printf("- Check: %s\n", settings.check ? "1" : "0");
}

/** The facts this binary and this machine report: the library version, and the capabilities
 *  compiled in and detected. */
struct machine_t {
    std::array<int, 3> version {FORKUNION_VERSION_MAJOR, FORKUNION_VERSION_MINOR, FORKUNION_VERSION_PATCH};
    capabilities_t compiled {};
    capabilities_t detected {};
};

/** Probes the capabilities @c machine_t reports. */
inline machine_t probe_machine() noexcept {
    machine_t machine;
    machine.compiled = comptime_capabilities();
    machine.detected = runtime_capabilities();
    return machine;
}

/** Prints @p label, then each bit of @p capabilities by its @c capability_name, comma-separated. */
inline void print_capabilities(char const *label, capabilities_t const capabilities) noexcept {
    std::printf("%s", label);
    char const *separator = "";
    for (unsigned int bit = 1; bit != 0; bit <<= 1) {
        char const *const name = capability_name(static_cast<capabilities_t>(bit));
        if (!(capabilities & bit) || !name) continue;
        std::printf("%s%s", separator, name);
        separator = ",";
    }
    std::printf("\n");
}

/** Prints the version line, then the capabilities as "- Compiled for:" and "- This machine:". */
inline void print(machine_t const &machine) noexcept {
    std::printf("ForkUnion %d.%d.%d\n", machine.version[0], machine.version[1], machine.version[2]);
    print_capabilities("- Compiled for: ", machine.compiled);
    print_capabilities("- This machine: ", machine.detected);
}

} // namespace ashvardanian::forkunion::bench
