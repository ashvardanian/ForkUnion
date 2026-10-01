/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief The settings, machine report and timed loop the C++ benchmarks share.
 *
 *  Both benchmarks read every variable once at start, and a value that does not parse prints one
 *  line and exits with status 1. Leaving a variable unset or empty keeps its default.
 *
 *  @verbatim
 *      Variable                           Default                  Meaning
 *      FORKUNION_SEED                     42                       Seed of every draw, or random
 *      FORKUNION_WARMUP                   1s                       Untimed run before timing
 *      FORKUNION_TIME_LIMIT               10s                      Timed window, like 500ms
 *      FORKUNION_THREADS                  all cores                Threads; 0 also means all cores
 *      FORKUNION_BACKEND                  forkunion_static_shared  Backend, per benchmark header
 *      FORKUNION_NBODY_COUNT              the thread count         Bodies to simulate
 *      FORKUNION_PROPAGATION_SCALE        14                       2^scale vertices per community
 *      FORKUNION_PROPAGATION_COMMUNITIES  64                       Communities strung on the ring
 *      FORKUNION_PROPAGATION_EDGE_FACTOR  16                       Raw edges per vertex
 *      FORKUNION_PROPAGATION_CHECK        false                    Also converge serially to check
 *  @endverbatim
 */
#pragma once
#include <cstdint> // `std::uint32_t`
#include <cstdio>  // `std::fprintf`, `std::printf`
#include <cstdlib> // `std::getenv`, `std::exit`

#include <array>        // `std::array`
#include <charconv>     // `std::from_chars`
#include <chrono>       // `std::chrono::steady_clock`, `std::chrono::milliseconds`
#include <optional>     // `std::optional`
#include <random>       // `std::random_device`
#include <string>       // `std::string`, `std::to_string`
#include <string_view>  // `std::string_view`
#include <system_error> // `std::errc`
#include <utility>      // `std::move`

#include "forkunion.hpp" // `FORKUNION_VERSION_*`, `capability_name`, `allowed_cores_count`

namespace fu = ashvardanian::forkunion;

namespace ashvardanian::forkunion::bench {

using steady_clock_t = std::chrono::steady_clock;
using time_point_t = steady_clock_t::time_point;

/** The text of the environment variable @p name, or nothing when it is unset or empty. */
inline std::optional<std::string_view> env_text(char const *name) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return std::nullopt;
    return std::string_view(text);
}

/** Parses the environment variable @p name with @p parse, or returns @p fallback when it is unset
 *  or empty. Text that does not parse prints `NAME="text" does not parse, expected <expected>` and
 *  exits with status 1, which leaves crash handlers quiet. */
template <typename value_type_, typename parse_type_>
[[nodiscard]] value_type_ env_parsed(char const *name, value_type_ fallback, parse_type_ &&parse,
                                     char const *expected) noexcept {
    std::optional<std::string_view> const text = env_text(name);
    if (!text) return fallback;
    if (std::optional<value_type_> value = parse(*text)) return *std::move(value);
    std::fprintf(stderr, "%s=\"%.*s\" does not parse, expected %s\n", name, static_cast<int>(text->size()),
                 text->data(), expected);
    std::exit(1);
}

/** A positive whole number, like "64". */
inline std::optional<std::size_t> parse_count(std::string_view text) noexcept {
    std::size_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || end != text.data() + text.size() || !count) return std::nullopt;
    return count;
}

/** A thread count like "8", or "0" for every core this process may run on. */
inline std::optional<std::size_t> parse_threads(std::string_view text) noexcept {
    if (text == "0") return allowed_cores_count();
    return parse_count(text);
}

/** A positive duration in whole milliseconds or seconds, like "200ms" or "10s". */
inline std::optional<std::chrono::milliseconds> parse_duration(std::string_view text) noexcept {
    std::uint32_t count = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), count);
    if (error != std::errc {} || !count) return std::nullopt;
    std::string_view const unit = text.substr(end - text.data());
    if (unit == "ms") return std::chrono::milliseconds(count);
    if (unit == "s") return std::chrono::seconds(count);
    return std::nullopt;
}

/** A 32-bit seed, or "random" for a fresh draw from @c std::random_device. */
inline std::optional<std::uint32_t> parse_seed(std::string_view text) noexcept {
    if (text == "random") return static_cast<std::uint32_t>(std::random_device {}());
    std::uint32_t seed = 0;
    auto const [end, error] = std::from_chars(text.data(), text.data() + text.size(), seed);
    if (error != std::errc {} || end != text.data() + text.size()) return std::nullopt;
    return seed;
}

inline std::size_t env_count(char const *name, std::size_t fallback) noexcept {
    return env_parsed(name, fallback, parse_count, "a positive count");
}

inline std::chrono::milliseconds env_duration(char const *name, std::chrono::milliseconds fallback) noexcept {
    return env_parsed(name, fallback, parse_duration, "a duration like 200ms or 10s");
}

inline bool env_flag(char const *name, bool fallback) noexcept {
    auto const parse_flag = [](std::string_view text) noexcept -> std::optional<bool> {
        if (text == "1" || text == "true") return true;
        if (text == "0" || text == "false") return false;
        return std::nullopt;
    };
    return env_parsed(name, fallback, parse_flag, "0, 1, true or false");
}

inline std::uint32_t env_seed(char const *name, std::uint32_t fallback) noexcept {
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random");
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
        return {name, calls_ / std::chrono::duration<double>(elapsed_).count(), counters_};
    }
};

/** Every benchmark setting, its default as the initializer, filled once by @c read_settings. */
struct settings_t {
    std::uint32_t seed = 42;
    std::chrono::milliseconds warmup = std::chrono::seconds(1);
    std::chrono::milliseconds time_limit = std::chrono::seconds(10);
    std::size_t threads = allowed_cores_count();
    std::string_view backend = "forkunion_static_shared";
    std::size_t bodies = threads;
    std::size_t scale = 14;
    std::size_t communities = 64;
    std::size_t edge_factor = 16;
    bool check = false;
};

/** Reads every @c settings_t variable, rejecting a zero count. */
inline settings_t read_settings() noexcept {
    settings_t settings;
    settings.seed = env_seed("FORKUNION_SEED", settings.seed);
    settings.warmup = env_duration("FORKUNION_WARMUP", settings.warmup);
    settings.time_limit = env_duration("FORKUNION_TIME_LIMIT", settings.time_limit);
    settings.threads = env_parsed("FORKUNION_THREADS", settings.threads, parse_threads, "a count, 0 for all cores");
    settings.backend = env_text("FORKUNION_BACKEND").value_or(settings.backend);
    settings.bodies = env_count("FORKUNION_NBODY_COUNT", settings.threads);
    settings.scale = env_count("FORKUNION_PROPAGATION_SCALE", settings.scale);
    settings.communities = env_count("FORKUNION_PROPAGATION_COMMUNITIES", settings.communities);
    settings.edge_factor = env_count("FORKUNION_PROPAGATION_EDGE_FACTOR", settings.edge_factor);
    settings.check = env_flag("FORKUNION_PROPAGATION_CHECK", settings.check);
    return settings;
}

/** Prints each setting as "- Name: value", in the grammar it parses from. */
inline void print(settings_t const &settings) {
    std::printf("- Seed: %u\n", static_cast<unsigned>(settings.seed));
    std::printf("- Warm-up: %s\n", spell_duration(settings.warmup).c_str());
    std::printf("- Time limit: %s\n", spell_duration(settings.time_limit).c_str());
    std::printf("- Threads: %zu\n", settings.threads);
    std::printf("- Backend: %.*s\n", static_cast<int>(settings.backend.size()), settings.backend.data());
    std::printf("- Bodies: %zu\n", settings.bodies);
    std::printf("- Scale: %zu\n", settings.scale);
    std::printf("- Communities: %zu\n", settings.communities);
    std::printf("- Edge factor: %zu\n", settings.edge_factor);
    std::printf("- Check: %s\n", settings.check ? "true" : "false");
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
