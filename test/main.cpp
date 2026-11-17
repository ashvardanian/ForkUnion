/**
 *  @file test/main.cpp
 *  @author Ash Vardanian
 *  @date May 2, 2025
 *  @brief Unit and stress tests for the pools, the harvested topology, and the parallel algorithms.
 *
 *  Every case is a @c static function reached through a @c run_test_case call in @c main, so a
 *  case absent from @c main does not run - it is the only index that cannot drift from what
 *  executes. Checks stay live whatever the build type, which is what the @c \#undef @c NDEBUG
 *  below buys: a failure reproducing only in Release is the one worth catching.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#include <cctype>  // `std::tolower`
#include <cmath>   // `std::llround`, `std::isfinite`
#include <csignal> // `std::signal`, `std::raise`
#include <cstddef> // `std::ptrdiff_t`
#include <cstdint> // `std::uint_least32_t`
#include <cstdio>  // `std::printf`, `std::fprintf`, `std::fputc`
#include <cstdlib> // `EXIT_FAILURE`, `std::getenv`, `std::exit`
#include <cstring> // `std::strcmp`, `std::strncmp`

#include <algorithm>       // `std::sort`, `std::max`
#include <atomic>          // `std::atomic`
#include <bit>             // `__cpp_lib_bit_cast`
#include <charconv>        // `std::from_chars`
#include <chrono>          // `std::chrono::steady_clock`
#include <compare>         // `operator<=>`
#include <format>          // `std::format_to`, `std::format_string`
#include <functional>      // `std::invoke`
#include <limits>          // `std::numeric_limits`
#include <random>          // `std::random_device`
#include <regex>           // `std::regex`, `std::regex_search`
#include <source_location> // `std::source_location`
#include <string_view>     // `std::string_view`
#include <system_error>    // `std::errc`
#include <thread>          // `std::thread`, `std::this_thread::yield`
#include <type_traits>     // `std::is_integral`, `std::is_enum`, `std::invoke_result_t`
#include <vector>          // `std::vector`

#include <forkunion.hpp>

/*  @c backtrace is glibc or Apple only; Bionic, FreeBSD and musl ship `<execinfo.h>` without it. */
#if FORKUNION_OS_POSIX_
#include <unistd.h> // `::write`, `STDERR_FILENO`
#if (FORKUNION_HAS_GLIBC_ || FORKUNION_OS_APPLE_) && __has_include(<execinfo.h>)
#include <execinfo.h> // `::backtrace`, `::backtrace_symbols_fd`
#define FORKUNION_TEST_WITH_BACKTRACE_ 1
#endif
#endif

namespace fu = ashvardanian::forkunion;

/** Stops in an attached debugger, or dumps core without one; SIGTRAP skips the crash handlers. */
inline void trap_into_debugger() noexcept {
#if FORKUNION_OS_WINDOWS_
    __debugbreak();
#elif FORKUNION_OS_POSIX_
    std::raise(SIGTRAP);
#else
    __builtin_trap(); // ? A WebAssembly sandbox has no signals to raise
#endif
}

/** Names the case in progress when the run crashes; defined once @c test_run exists. */
inline void install_test_signal_handlers() noexcept;

/** Output iterator handing each character straight to a @c std::FILE. */
struct file_output_iterator_t {
    using difference_type = std::ptrdiff_t;

    std::FILE *stream {};

    file_output_iterator_t &operator*() noexcept { return *this; }
    file_output_iterator_t &operator++() noexcept { return *this; }
    file_output_iterator_t operator++(int) noexcept { return *this; }
    file_output_iterator_t &operator=(char character) noexcept {
        std::fputc(character, stream);
        return *this;
    }
};

/** Writes one line to @p stream, formatted from @p format and @p args, then its newline. */
template <typename... args_types_>
void print_line(std::FILE *stream, std::format_string<args_types_...> format, args_types_ &&...args) {
    std::format_to(file_output_iterator_t {stream}, format, std::forward<args_types_>(args)...);
    std::fputc('\n', stream);
}

#pragma region Shared Settings

/** Exits with one line naming a setting that does not parse, so a bad value never runs anything. */
[[noreturn]] inline void exit_unparsed(char const *variable, char const *text, char const *expected) noexcept {
    print_line(stderr, "{}=\"{}\" does not parse, expected {}", variable, text, expected);
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), count);
    std::string_view const unit = spelled.substr(static_cast<std::size_t>(end - spelled.data()));
    if (parse_status == std::errc {} && count != 0) {
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
    auto const [end, parse_status] = std::from_chars(spelled.data(), spelled.data() + spelled.size(), seed);
    if (parse_status != std::errc {} || end != spelled.data() + spelled.size())
        exit_unparsed(variable, text, "an unsigned integer or random");
    return seed;
}

/** Which labels a run selects: those its regex matches somewhere, every label while unset. */
struct regex_filter_t {
    std::string_view text;
    std::regex regex {""};

    bool selects(std::string_view label) const noexcept { return std::regex_search(label.begin(), label.end(), regex); }
};

/** The filter a setting spells, compiled once; one that does not compile exits as a bad setting. */
inline regex_filter_t read_regex_filter_setting(int argc, char **argv, char const *variable,
                                                std::string_view flag) noexcept {
    regex_filter_t filter;
    char const *text = setting_text(argc, argv, variable, flag);
    if (!text) return filter;
    filter.text = text;
#if defined(__cpp_exceptions) && __cpp_exceptions
    try {
        filter.regex.assign(filter.text.begin(), filter.text.end());
    }
    catch (std::regex_error const &) {
        exit_unparsed(variable, text, "a regular expression");
    }
#else
    filter.regex.assign(filter.text.begin(), filter.text.end());
#endif
    return filter;
}

/** What every suite and bench reads before its own settings; @c program starts the rerun lines. */
struct common_settings_t {
    std::uint64_t seed = 42;
    regex_filter_t regex_filter;
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
 *  @brief SplitMix64 over one 64-bit counter, seeded by a @ref stream_key , bit-identical to the
 *      JavaScript generator NumKong ships.
 *
 *  Distributions are not specified to draw the same sequence on every standard library, so a
 *  failure would stop reproducing from its seed when the toolchain changed; @ref below is the
 *  portable bounded draw.
 */
struct split_mix_generator_t {
    using result_type = std::uint64_t;
    std::uint64_t counter = 0;

    explicit split_mix_generator_t(std::uint64_t key) noexcept : counter(key) {}
    static constexpr result_type(min)() noexcept { return 0; }
    static constexpr result_type(max)() noexcept { return ~result_type {0}; }
    result_type operator()() noexcept { return split_mix_bits(counter += 0x9E3779B97F4A7C15ull); }

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

/** What this suite reads: the common settings and the run's own, with nothing beyond them. */
struct test_settings_t : common_settings_t {
    double scale = 1.0;
    bool trap_on_failure = false;
    bool refuse_skips = false;
};

/** Reads every @c test_settings_t field, each from its flag or else its variable. */
inline test_settings_t read_test_settings(int argc, char **argv) noexcept {
    test_settings_t settings;
    settings.seed = read_seed_setting(argc, argv, "FORKUNION_SEED", "--seed");
    settings.regex_filter = read_regex_filter_setting(argc, argv, "FORKUNION_FILTER", "--filter");
    settings.scale = read_real_setting(argc, argv, "FORKUNION_SCALE", "--scale", 1.0);
    settings.trap_on_failure = read_flag_setting(argc, argv, "FORKUNION_TRAP_ON_FAILURE", "--trap-on-failure");
    settings.refuse_skips = read_flag_setting(argc, argv, "FORKUNION_NO_SKIPS", "--no-skips");
    settings.program = argc != 0 ? argv[0] : "";
    return settings;
}

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/** The machine the colocated and distributed cases spawn on, harvested once by @c probe_machine. */
static fu::machine_topology_t &machine_topology() noexcept {
    static fu::machine_topology_t topology;
    return topology;
}
#endif

/** Prints @p label, then each bit of @p capabilities by its @c capability_name, comma-separated. */
static void print_capabilities(char const *label, fu::capabilities_t const capabilities) noexcept {
    std::printf("%s", label);
    char const *separator = "";
    for (unsigned int bit = 1; bit != 0; bit <<= 1) {
        char const *const name = fu::capability_name(static_cast<fu::capabilities_t>(bit));
        if (!(capabilities & bit) || !name) continue;
        std::printf("%s%s", separator, name);
        separator = ",";
    }
    std::printf("\n");
}

/** What this binary was compiled for and what this machine offers. */
struct machine_t {
    fu::capabilities_t compiled {};
    fu::capabilities_t detected {};
};

/** Probes the capabilities and harvests the @c machine_topology the colocated cases spawn on. */
static machine_t probe_machine() noexcept {
#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN
    if (fu::failed(machine_topology().harvest())) {
        std::fprintf(stderr, "Failed to harvest the machine topology\n");
        std::exit(EXIT_FAILURE);
    }
#endif
    return {fu::comptime_capabilities(), fu::runtime_capabilities()};
}

/** Prints the version, the capabilities compiled in and detected, and the harvested NUMA layout. */
static void print(machine_t const &machine) noexcept {
    std::printf("ForkUnion %d.%d.%d\n", FORKUNION_VERSION_MAJOR, FORKUNION_VERSION_MINOR, FORKUNION_VERSION_PATCH);
    print_capabilities("- Compiled for: ", machine.compiled);
    print_capabilities("- This machine: ", machine.detected);
#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN
    fu::log_numa_topology_t {}(machine_topology(), fu::logging_colors_t {});
#endif
}

#pragma region Shared Run

/** What a case reports beyond its checks: that it ran, or why it skipped. */
struct case_result_t {
    char const *skip_reason = nullptr;

    [[nodiscard]] static constexpr case_result_t skipped(char const *reason) noexcept { return {reason}; }
};

/**
 *  @brief The run in progress: the harness's one sanctioned mutable global.
 *
 *  Checks fire on worker threads and inside helpers as well as in a case's own body, so the run
 *  counts failed checks in one place and @ref run_test_case diffs that count around each case,
 *  rather than threading a tally through every call. After a crash, printing @c test_run from a
 *  core dump names the case in progress and the last failed check, and the signal handlers print
 *  the case too.
 */
struct test_run_t {
    test_settings_t settings {};
    char const *current_case = nullptr;
    std::atomic<std::size_t> failed_checks {0};
    std::atomic<char const *> last_failed_file {nullptr};
    std::atomic<std::uint_least32_t> last_failed_line {0};
    std::size_t passed_cases = 0;
    std::size_t failed_cases = 0;
    std::size_t skipped_cases = 0;
};

inline test_run_t test_run;

/** How long any wait in a test may take; a passing case never waits it out, so it bounds hangs. */
inline constexpr std::chrono::seconds wait_limit_k {120};

/** Yields until @p predicate holds; false when @ref wait_limit_k passed first. */
template <typename predicate_type_>
bool spin_until(predicate_type_ &&predicate) noexcept {
    auto const deadline = std::chrono::steady_clock::now() + wait_limit_k;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

/** @p baseline iterations under the @c SCALE setting, never fewer than one. */
inline std::size_t scaled_iterations(std::size_t baseline) noexcept {
    double const scaled = static_cast<double>(baseline) * test_run.settings.scale;
    return scaled < 1 ? 1 : static_cast<std::size_t>(std::llround(scaled));
}

/** The step that visits a @p coverage fraction of a sweep, at least one. */
inline std::size_t sweep_stride(double coverage) noexcept {
    double const stride = 1.0 / coverage;
    return stride < 1 ? 1 : static_cast<std::size_t>(std::llround(stride));
}

/** The @p index -th stream of the case in progress, so a filtered run draws what a full one did. */
inline std::uint64_t case_stream_key(std::uint64_t index = 0) noexcept {
    return stream_key(test_run.settings.seed, test_run.current_case ? test_run.current_case : "", index);
}

/** Line-buffers stdout, prints the machine, reads the settings, arms the crash handlers and
 *  prints the settings line, before the first case. */
inline void start_test_run(int argc, char **argv) noexcept {
    std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ); // before any output, or the C library keeps its buffering
    print(probe_machine());
    test_run.settings = read_test_settings(argc, argv);
    install_test_signal_handlers();
    print_line(stdout, "- Seed: {}, scale: {}", test_run.settings.seed, test_run.settings.scale);
    print_line(
        stdout, "- Filter: {}",
        test_run.settings.regex_filter.text.empty() ? std::string_view("none") : test_run.settings.regex_filter.text);
}

/** Counts one failed check at @p where , and traps when the run asked to stop at the first. */
inline void record_failed_check(std::source_location where) noexcept {
    std::fflush(stdout); // so a log that merges both streams keeps the failure under its case
    print_line(stderr, "  check failed at {}:{}", where.file_name(), where.line());
    test_run.last_failed_file.store(where.file_name(), std::memory_order_relaxed);
    test_run.last_failed_line.store(where.line(), std::memory_order_relaxed);
    test_run.failed_checks.fetch_add(1, std::memory_order_relaxed);
    if (test_run.settings.trap_on_failure) trap_into_debugger();
}

/** Holds when @p is_satisfied does; otherwise counts a failed check at the caller's line. */
inline bool expect_true(bool is_satisfied, std::source_location where = std::source_location::current()) noexcept {
    if (!is_satisfied) record_failed_check(where);
    return is_satisfied;
}

/** A case label spelled as a regex that matches only itself, every metacharacter escaped. */
struct regex_escaped_t {
    std::string_view label;
};

/** Writes a @ref regex_escaped_t straight to the output, a backslash before each metacharacter. */
template <>
struct std::formatter<regex_escaped_t> {
    constexpr auto parse(std::format_parse_context &context) { return context.begin(); }
    template <typename context_type_>
    auto format(regex_escaped_t const &escaped, context_type_ &context) const {
        auto output = context.out();
        for (char const character : escaped.label) {
            if (std::string_view {"\\^$.|?*+()[]{}"}.find(character) != std::string_view::npos) *output++ = '\\';
            *output++ = character;
        }
        return output;
    }
};

/** Prints one case's verdict, and the line that reruns it alone when it failed. */
inline void report_test_case(char const *label, std::size_t failed_checks, char const *skip_reason,
                             double seconds) noexcept {
    if (skip_reason) {
        print_line(stdout, "- {} ... skipped: {}", label, skip_reason);
        return;
    }
    if (failed_checks == 0) {
        print_line(stdout, "- {} ... ok ({:.2f} s)", label, seconds);
        return;
    }
    std::fflush(stdout);
    print_line(stderr, "- {} ... FAILED: {} failed check(s)", label, failed_checks);
    print_line(stderr, "  rerun: {} --seed {} --filter '^{}$'", test_run.settings.program, test_run.settings.seed,
               regex_escaped_t {label});
}

/** Runs one case unless the filter leaves it out, and records whether it passed or skipped. */
template <typename case_function_type_, typename... argument_types_>
void run_test_case(char const *label, case_function_type_ &&case_function, argument_types_ &&...arguments) noexcept {
    if (!test_run.settings.regex_filter.selects(label)) return;
    test_run.current_case = label;
    std::size_t const failed_before = test_run.failed_checks.load();
    auto const started = std::chrono::steady_clock::now();
    case_result_t result {};
    using returned_type = std::invoke_result_t<case_function_type_, argument_types_...>;
    if constexpr (std::is_void_v<returned_type>)
        std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    else result = std::invoke(case_function, std::forward<argument_types_>(arguments)...);
    std::chrono::duration<double> const elapsed = std::chrono::steady_clock::now() - started;
    std::size_t const failed_checks = test_run.failed_checks.load() - failed_before;
    if (result.skip_reason) ++test_run.skipped_cases;
    else if (failed_checks != 0) ++test_run.failed_cases;
    else ++test_run.passed_cases;
    report_test_case(label, failed_checks, result.skip_reason, elapsed.count());
    test_run.current_case = nullptr;
}

/** The exit code: 1 when a case or a check outside any case failed, or nothing ran; 3 when
 *  skips are refused and one happened; else 0. */
inline int finish_test_run() noexcept {
    print_line(stdout, "{} passed, {} failed, {} skipped, seed {}", test_run.passed_cases, test_run.failed_cases,
               test_run.skipped_cases, test_run.settings.seed);
    bool const failed = test_run.failed_cases != 0 || test_run.failed_checks.load() != 0;
    if (failed || test_run.passed_cases + test_run.skipped_cases == 0) return 1;
    if (test_run.skipped_cases != 0 && test_run.settings.refuse_skips) return 3;
    return 0;
}

#pragma endregion Shared Run

/** Writes @p text to stderr without stdio, whose lock the crashing thread may hold. */
static void write_to_stderr_([[maybe_unused]] std::string_view text) noexcept {
#if FORKUNION_OS_WINDOWS_
    DWORD written = 0;
    ::WriteFile(::GetStdHandle(STD_ERROR_HANDLE), text.data(), static_cast<DWORD>(text.size()), &written, nullptr);
#elif FORKUNION_OS_POSIX_
    [[maybe_unused]] ssize_t const written = ::write(STDERR_FILENO, text.data(), text.size());
#endif
}

/** Names what ended the run and the case in progress, as a banner with the signal and the label. */
static void write_crash_banner_(std::string_view what) noexcept {
    write_to_stderr_("\n*** ");
    write_to_stderr_(what);
    write_to_stderr_(" in case '");
    write_to_stderr_(test_run.current_case ? test_run.current_case : "none");
    write_to_stderr_("' - backtrace follows ***\n");
}

#if FORKUNION_OS_POSIX_

/** Names the signal and the case in progress, prints a backtrace, then re-raises for the core. */
extern "C" void on_fatal_signal_(int signal_number) noexcept {
    std::string_view name = "SIGSEGV";
    if (signal_number == SIGABRT) name = "SIGABRT";
    else if (signal_number == SIGILL) name = "SIGILL";
    else if (signal_number == SIGFPE) name = "SIGFPE";
    else if (signal_number == SIGBUS) name = "SIGBUS";
    write_crash_banner_(name);
#if FORKUNION_TEST_WITH_BACKTRACE_
    void *frames[64];
    ::backtrace_symbols_fd(frames, ::backtrace(frames, 64), STDERR_FILENO);
#endif
    std::signal(signal_number, SIG_DFL);
    std::raise(signal_number);
}

inline void install_test_signal_handlers() noexcept {
    for (int signal_number : {SIGSEGV, SIGABRT, SIGFPE, SIGILL, SIGBUS}) std::signal(signal_number, on_fatal_signal_);
}

#elif FORKUNION_OS_WINDOWS_

/** Names the case in progress before a structured exception, such as a bad access, ends the run. */
static LONG WINAPI on_unhandled_exception_(EXCEPTION_POINTERS *) noexcept {
    write_crash_banner_("unhandled exception");
    return EXCEPTION_CONTINUE_SEARCH; // ? The default handler, or an attached debugger, still sees it
}

inline void install_test_signal_handlers() noexcept { ::SetUnhandledExceptionFilter(on_unhandled_exception_); }

#else
inline void install_test_signal_handlers() noexcept {}
#endif

/** Whether a failed comparison can name @p type_ : an enum, a result holding one, or anything
 *  @c std::format takes. */
template <typename type_>
constexpr bool printable_() noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_enum_v<bare_t>) return true;
    else if constexpr (requires(bare_t const &value) {
                           value.status();
                           value.has_value();
                           *value;
                       })
        return printable_<decltype(*std::declval<bare_t const &>())>();
    else return std::is_default_constructible_v<std::formatter<bare_t, char>>;
}

/** Prints one side of a failed comparison, an enum, such as a status, by its underlying value. */
template <typename type_>
void print_operand_(type_ const &value) noexcept {
    using bare_t = std::remove_cvref_t<type_>;
    if constexpr (std::is_enum_v<bare_t>)
        std::format_to(file_output_iterator_t {stderr}, "{}", static_cast<std::underlying_type_t<bare_t>>(value));
    else if constexpr (requires {
                           value.status();
                           value.has_value();
                           *value;
                       }) {
        // A result that holds something was compared for what it holds, so that is what a reader
        // needs; the status is the answer only when there is no value to have compared.
        if (!value.has_value()) print_operand_(value.status());
        else print_operand_(*value);
    }
    else std::format_to(file_output_iterator_t {stderr}, "{}", value);
}

/** Prints both sides of a failed equality under its location line, when both can be named. */
template <typename actual_type_, typename wanted_type_>
void print_unequal_([[maybe_unused]] actual_type_ const &actual, [[maybe_unused]] wanted_type_ const &wanted) noexcept {
    if constexpr (printable_<actual_type_>() && printable_<wanted_type_>()) {
        std::fputs("    expected ", stderr);
        print_operand_(wanted);
        std::fputs(", got ", stderr);
        print_operand_(actual);
        std::fputc('\n', stderr);
    }
}

/** Holds when @p actual equals @p wanted ; otherwise counts a failed check and prints both. */
template <typename actual_type_, typename wanted_type_>
static bool expect_equal(actual_type_ const &actual, wanted_type_ const &wanted,
                         std::source_location where = std::source_location::current()) noexcept {
    if (actual == wanted) return true;
    record_failed_check(where);
    print_unequal_(actual, wanted);
    return false;
}

using fu32_t = fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                             fu::preferred_cache_hints_t, std::uint32_t>;
using fu16_t = fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                             fu::preferred_cache_hints_t, std::uint16_t>;
using fu8_t = fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                            fu::preferred_cache_hints_t, std::uint8_t>;

/*  Explicitly instantiate the thread-pools to cover all of their logic, but avoid the "duplicate
 *  explicit instantiation" error on platforms where @c std::size_t is @c uint64_t.
 *
 *  template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, std::size_t> */
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                             fu::preferred_cache_hints_t, std::uint32_t>;
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                             fu::preferred_cache_hints_t, std::uint16_t>;
template class fu::flat_pool<std::allocator<std::thread>, fu::standard_yield_t, fu::standard_atomic_ref,
                             fu::preferred_cache_hints_t, std::uint8_t>;

template class fu::spin_mutex<fu::preferred_yield_t>;

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN
template struct fu::colocated_pool<>;
template struct fu::distributed_pool<>;
#endif

/**
 *  @brief Exhausts every tasks-and-threads pair in the index space to verify a clean partition.
 *
 *  Every task index must fall in bounds and be visited exactly once across the per-thread
 *  subranges; a gap or overlap would mean lost or double-dispatched work in any static scheduler.
 */
template <typename index_type_ = std::uint8_t>
void test_indexed_split_tiles_the_range() noexcept {
    std::size_t max_tasks = (std::numeric_limits<index_type_>::max)();
    std::size_t max_threads = (std::numeric_limits<index_type_>::max)();
    std::vector<bool> visits(max_tasks);

    for (std::size_t threads = 1; threads < max_threads; ++threads) {
        for (std::size_t tasks = 0; tasks < max_tasks; ++tasks) {

            // Reset visits for each test case
            std::fill_n(visits.begin(), max_tasks, false);

            fu::indexed_split<index_type_> split {static_cast<index_type_>(tasks), static_cast<index_type_>(threads)};
            for (std::size_t thread = 0; thread < threads; ++thread) {
                auto subrange = split[static_cast<index_type_>(thread)];
                for (std::size_t task = subrange.first; task < subrange.first + subrange.count; ++task) {
                    if (!expect_true(task < tasks)) return; // ! Out of bounds, so the mark below would overrun
                    expect_true(!visits[task]);             // Not already visited
                    visits[task] = true;                    // Mark as visited
                }
            }
        }
    }
}

/**
 *  @brief Exhausts every start-end-seed triple in the index space, checking each walk permutes.
 *
 *  Counting alone can't tell a permutation from a walk that revisits values and skips others, so
 *  each value's first visit is recorded - a revisit drains some victim twice.
 */
template <typename index_type_ = std::uint8_t>
void test_coprime_permutation_visits_each_value_once() noexcept {
    constexpr std::size_t max_tasks = (std::numeric_limits<index_type_>::max)();
    std::vector<bool> visits(max_tasks);

    for (std::size_t start = 0; start < max_tasks; ++start) {
        for (std::size_t end = start + 1; end < max_tasks; ++end) {
            for (std::size_t seed = 0; seed < max_tasks; ++seed) {

                // Create a coprime permutation and make sure it only covers the range [start, end)
                index_type_ const range_size = static_cast<index_type_>(end - start);
                fu::coprime_permutation_range<index_type_> permutation(static_cast<index_type_>(start), range_size,
                                                                       static_cast<index_type_>(seed));

                // Reset visits for each test case
                std::fill_n(visits.begin(), max_tasks, false);

                // Counting alone can't tell a permutation from a walk that revisits values and
                // skips others - a revisit makes a work-stealing thread drain the same victim
                // twice, overrunning its cursor; each value must appear exactly once.
                std::size_t count_matches = 0;
                for (auto value : permutation) {
                    // ! Out of range, so the mark below would overrun
                    if (!expect_true(value >= start) || !expect_true(value < end)) return;

                    std::size_t const offset = static_cast<std::size_t>(value) - start;
                    expect_true(!visits[offset]); // ! Revisited a value, so this is not a permutation

                    visits[offset] = true;
                    count_matches++;
                }
                expect_equal(count_matches, range_size); // Every value in the range was covered
            }
        }
    }
}

/**
 *  @brief Checks that a harvested topology is internally consistent, on whatever host runs it.
 *
 *  Deliberately @b not gated on @c FORKUNION_WITH_PLACE_MEMORY_ON_DOMAIN: some platforms harvest a
 *  topology without compiling the NUMA pools, so gating this on the pools leaves their harvest
 *  wholly untested. A host with no harvest at all reports @c false and is skipped rather than
 *  failed - the absence of a topology is not a broken topology.
 */
static case_result_t test_topology_counts_and_core_partition() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest()))
        return case_result_t::skipped("fu::machine_topology_t::harvest is refused on this machine");

    std::size_t const compute_domains = topology.compute_domains_count();
    std::size_t const memory_domains = topology.memory_domains_count();
    std::size_t const compute_levels = topology.compute_levels_count();
    expect_true(compute_domains != 0);
    expect_true(memory_domains != 0);
    expect_true(compute_levels != 0);

    // Levels are dense ranks over domains, so they can never outnumber the domains they rank.
    expect_true(compute_levels <= compute_domains);

    // Levels are a *dense* rank, so each of [0, levels) must be claimed by at least one domain.
    // Merely staying in range is too weak - it would accept a count inflated past distinct levels.
    std::vector<bool> compute_level_seen(compute_levels, false);

    std::size_t cores_across_domains = 0;
    for (std::size_t i = 0; i < compute_domains; ++i) {
        fu::compute_domain_t const &domain = topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(i));
        expect_true(!domain.core_ids.empty());
        expect_true(domain.memory_domain_index < memory_domains);
        if (!expect_true(domain.compute_level < compute_levels)) return {}; // ! The rank below indexes by it
        compute_level_seen[domain.compute_level] = true;
        cores_across_domains += domain.core_ids.size();
    }
    // Every core must belong to exactly one compute domain.
    expect_equal(cores_across_domains, topology.logical_cores_count());

    for (std::size_t level = 0; level < compute_levels; ++level)
        expect_true(compute_level_seen[level]); // ? An unclaimed rank means the count is inflated
    return {};
}

constexpr std::size_t default_parallel_tasks_k = 10000; // 10K

/*  The maker vocabulary the whole three-tier test table hangs on: every templated test constructs
 *  its pool through a maker and spawns it on `maker.scope()`, so the same body runs against the
 *  flat, colocated, and distributed pools without knowing which one it drives.  */

/** Makes @c flat_pool_ts scoped to a thread count - the allowed cores, times oversubscription. */
struct make_pool_t {
    fu::flat_pool_t construct() const noexcept { return fu::flat_pool_t(); }
    std::size_t scope(std::size_t oversubscription = 1) const noexcept {
        return fu::allowed_cores_count() * oversubscription;
    }
};

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/** Makes @c colocated_pool_ts scoped to the machine's first compute domain. */
struct make_colocated_pool_t {
    fu::colocated_pool_t construct() const noexcept { return fu::colocated_pool_t("forkunion"); }
    fu::compute_domain_t scope(std::size_t = 0) const noexcept {
        return machine_topology().compute_domain_at(fu::compute_domain_index_t {});
    }
};

/** Makes @c distributed_pool_ts scoped to the whole harvested topology. */
struct make_distributed_pool_t {
    fu::distributed_pool_t construct() const noexcept { return fu::distributed_pool_t("forkunion"); }
    fu::machine_topology_t const &scope(std::size_t = 0) const noexcept { return machine_topology(); }
};
#endif

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/**
 *  @brief The tier derivation must rank a synthetic edge log by bandwidth first, latency second.
 *
 *  A tier is a property of the medium: HBM splits from DDR on bandwidth despite a worse latency,
 *  two DDR sockets within both jitter bands share a tier, a same-bandwidth medium past the 1.5x
 *  latency band splits, CXL trails on either key, an unobserved domain lands one tier past the
 *  slowest, and a worse duplicate recording must not move its domain - the best per metric wins.
 */
static void test_fabric_ranks_bandwidth_before_latency() noexcept {
    fu::measured_edge_t const edges[] = {
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {0}, 110, 3'000'000}, // ? HBM: 3 TB/s
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {1}, 100, 300'000},   // ? DDR, faster latency
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {2}, 104, 290'000},   // ? DDR, second socket
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {3}, 250, 100'000},   // ? CXL expander
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {0}, 400, 500'000},   // ? Worse repeat, ignored
        {fu::compute_domain_index_t {0}, fu::memory_domain_index_t {5}, 170, 285'000},   // ? DDR-wide, 1.7x latency
        // ? Domain 4 stays unobserved - a cpuless target no worker could first-touch
    };
    std::size_t levels[6];
    std::size_t scratch[24];
    std::size_t const count = fu::derive_memory_levels_(edges, sizeof(edges) / sizeof(edges[0]), levels, 6, scratch);

    expect_equal(count, std::size_t(5));     // HBM, DDR, the laggard, CXL, and the unobserved tier
    expect_equal(levels[0], std::size_t(0)); // Bandwidth ranks HBM first despite the worse latency
    expect_equal(levels[1], std::size_t(1)); // DDR opens the next tier
    expect_equal(levels[2], std::size_t(1)); // The second socket sits within both jitter bands
    expect_equal(levels[5], std::size_t(2)); // Same bandwidth band, but latency past 1.5x splits
    expect_equal(levels[3], std::size_t(3)); // CXL trails on both keys
    expect_equal(levels[4], std::size_t(4)); // Unobserved shares one tier past the slowest
}

/** A harvested fabric must cover every reachable edge with sane bounds and leave unreachable ones
 *  unwalked. No local-beats-remote assertion on purpose: emulated-NUMA guests legitimately measure
 *  every edge the same. */
static void test_fabric_harvest_fills_edges() noexcept {
    fu::machine_topology_t const &topology = machine_topology();
    alignas(fu::default_alignment_k) fu::distributed_pool<> pool;
    if (!expect_true(fu::succeeded(pool.spawn(topology)))) return;

    fu::measured_fabric_t fabric;
    expect_true(fabric.memory_latency(fu::compute_domain_index_t {}, fu::memory_domain_index_t {}) == 0);
    expect_true(fabric.memory_levels_count() == 1); // ? Unharvested: zeros everywhere, one tier
    expect_true(fu::succeeded(fabric.harvest(topology, pool)));
    expect_equal(fabric.compute_domains_count(), topology.compute_domains_count());
    expect_equal(fabric.memory_domains_count(), topology.memory_domains_count());

    for (std::size_t domain = 0; domain != pool.compute_domains_count(); ++domain) {
        fu::compute_domain_index_t const initiator = static_cast<fu::compute_domain_index_t>(domain);
        fu::memory_domain_index_t const local = topology.local_memory_of(initiator);
        expect_true(fabric.memory_distance(initiator, local) == 10); // The local edge anchors the SLIT scale

        for (std::size_t target = 0; target != topology.memory_domains_count(); ++target) {
            fu::memory_domain_index_t const to = static_cast<fu::memory_domain_index_t>(target);
            std::size_t const measured = fabric.memory_latency(initiator, to);

            // The harvest only walks edges some pool worker can first-touch, so a cpuless domain -
            // a CXL expander, or a node whose cores sit outside the pool - stays at zero and
            // answers with the unmeasured-remote fallback.
            bool reachable = false;
            for (std::size_t other = 0; other != pool.compute_domains_count() && !reachable; ++other)
                reachable =
                    topology.compute_domain_at(static_cast<fu::compute_domain_index_t>(other)).memory_domain_index ==
                    to;
            if (!reachable) {
                expect_true(measured == 0);
                expect_true(fabric.memory_distance(initiator, to) == 20);
                continue;
            }

            expect_true(measured > 0);       // Every reachable edge was walked
            expect_true(measured < 100'000); // A dependent load is nanoseconds, not a tenth of a millisecond

            std::size_t const streamed = fabric.memory_bandwidth(initiator, to);
            expect_true(streamed > 0);           // Every reachable edge was streamed
            expect_true(streamed < 100'000'000); // 100 TB/s exceeds any fabric - and catches an elided sum

            expect_true(fabric.memory_distance(initiator, to) >= 10); // Local is the row's floor
        }
    }

    // The derived tiers are dense ordinals: some domain is tier 0, none reaches the count.
    bool some_domain_is_fastest = false;
    for (std::size_t target = 0; target != topology.memory_domains_count(); ++target) {
        std::size_t const level = fabric.memory_level_in(static_cast<fu::memory_domain_index_t>(target));
        expect_true(level < fabric.memory_levels_count());
        some_domain_is_fastest |= level == 0;
    }
    expect_true(some_domain_is_fastest);

    // Bulk-snapshot semantics: a second harvest replaces the first and stays sane.
    expect_true(fu::succeeded(fabric.harvest(topology, pool)));
    expect_true(fabric.memory_latency(fu::compute_domain_index_t {},
                                      topology.local_memory_of(fu::compute_domain_index_t {})) > 0);
}
#endif // FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/** Zero threads is not a pool, nor is a count that wraps its size: both are refused cleanly. */
static void test_spawn_rejects_zero_threads() noexcept {
    fu::flat_pool_t pool;
    expect_true(fu::failed(pool.spawn(0u)));
    expect_equal(pool.spawn((std::numeric_limits<std::size_t>::max)()), fu::status_t::bad_alloc_k);
}

/** The default spawn - one thread per allowed core - must succeed on every pool type. */
template <typename make_pool_type_ = make_pool_t>
static void test_spawn_succeeds_at_full_width() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;
    expect_equal(pool.spawn(maker.scope()), fu::status_t::already_spawned_k);
}

/** The pool is the single source of truth for exclusivity, even across re-spawns. */
template <typename make_pool_type_ = make_pool_t>
static void test_caller_exclusivity_is_queried_not_cached() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();

    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;
    expect_equal(pool.caller_exclusivity(), fu::caller_inclusive_k);

    // Re-spawn with the opposite mode: the query must follow, not a stale cache
    pool.terminate();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;
    expect_equal(pool.caller_exclusivity(), fu::caller_exclusive_k);
}

/** Make sure that @c for_threads is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_threads_reaches_every_worker() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    pool.for_threads([&](std::size_t const thread_index) noexcept { //
        visited[thread_index].store(true, std::memory_order_relaxed);
    });

    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect_true(visited[i]);
}

/** Make sure that @c unsafe_for_threads is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_unsafe_for_threads_reaches_every_worker() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    auto on_each_thread = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };
    pool.unsafe_for_threads(on_each_thread);
    pool.unsafe_join();

    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect_true(visited[i]);
}

/** Tests generation token polling with two caller-exclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_polling_on_an_exclusive_pool() noexcept {

    auto maker = make_pool_type_ {};
    auto pool_a = maker.construct();
    auto pool_b = maker.construct();

    // Polling before join is the caller-exclusive pattern: on inclusive pools the caller owes a
    // slice that only runs inside `unsafe_join`.
    if (!expect_true(fu::succeeded(pool_a.spawn(maker.scope(), fu::caller_exclusive_k)))) return;
    if (!expect_true(fu::succeeded(pool_b.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

    std::vector<std::atomic<bool>> visited_a(pool_a.threads_count());
    std::vector<std::atomic<bool>> visited_b(pool_b.threads_count());

    auto work_a = [&](std::size_t const thread_index) noexcept {
        visited_a[thread_index].store(true, std::memory_order_relaxed);
    };
    auto work_b = [&](std::size_t const thread_index) noexcept {
        visited_b[thread_index].store(true, std::memory_order_relaxed);
    };

    // Drive one pool through the raw token API and the other through the RAII guard
    auto generation_a = pool_a.unsafe_for_threads(work_a);
    auto broadcast_b = pool_b.for_threads(work_b); // ? Dispatches at construction

    // Poll both pools until complete
    bool a_done = false, b_done = false;
    expect_true(spin_until([&] {
        if (!a_done) a_done = pool_a.is_complete(generation_a);
        if (!b_done) b_done = broadcast_b.is_complete();
        return a_done && b_done;
    }));

    pool_a.unsafe_join(generation_a);
    broadcast_b.join();

    for (std::size_t i = 0; i < pool_a.threads_count(); ++i) expect_true(visited_a[i]);
    for (std::size_t i = 0; i < pool_b.threads_count(); ++i) expect_true(visited_b[i]);
}

/** On an exclusive pool thread 0 forks from inside its slice three levels deep while the other
 *  threads' forks and a rival dispatch are refused; the caller polls the outer generation, a second
 *  join of an inner fork does nothing, and the pool takes a plain dispatch once the outer ends. */
template <typename make_pool_type_ = make_pool_t>
static void test_forks_from_thread_zero() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;
    std::size_t const threads = pool.threads_count();

    constexpr std::size_t tasks_k = 1000;
    struct visits_t {
        std::vector<std::atomic<std::size_t>> first, second, third;
        std::atomic<std::size_t> refused {0}, rival_checked {0}, third_forks {0};
    } visits {std::vector<std::atomic<std::size_t>>(tasks_k), std::vector<std::atomic<std::size_t>>(tasks_k),
              std::vector<std::atomic<std::size_t>>(threads)};

    auto third_level = [&](std::size_t thread) noexcept { visits.third[thread].fetch_add(1); };
    auto second_level = [&](std::size_t task, auto at) noexcept {
        visits.second[task].fetch_add(1);
        if (at.thread == 0 && visits.third_forks.fetch_add(1) == 0) pool.for_threads(third_level);
    };
    auto first_level = [&](std::size_t task, auto) noexcept {
        visits.first[task].fetch_add(1);
        if (task != 0) return; // ? Static slicing hands task 0 to thread 0
        // Joined twice, as the C interface joins; the second join of thread 0's fork does nothing
        auto twice_joined = pool.for_n_dynamic(tasks_k, second_level);
        twice_joined.join();
    };
    auto outer = [&](std::size_t thread) noexcept {
        if (thread != 0) {
            auto refused = [](std::size_t) noexcept {};
            if (pool.unsafe_for_threads(refused) == 0) visits.refused.fetch_add(1);
            return;
        }
        expect_true(spin_until([&] { return visits.rival_checked.load(std::memory_order_acquire) != 0; }));
        pool.for_n(tasks_k, first_level);
    };

    auto const generation = pool.unsafe_for_threads(outer);
    expect_true(generation != 0);
    auto rival = [](std::size_t) noexcept {};
    expect_true(pool.unsafe_for_threads(rival) == 0);
    visits.rival_checked.store(1, std::memory_order_release);
    expect_true(spin_until([&] { return pool.is_complete(generation); }));
    pool.unsafe_join(generation);

    for (std::size_t task = 0; task != tasks_k; ++task) expect_true(visits.first[task].load() == 1);
    for (std::size_t task = 0; task != tasks_k; ++task) expect_true(visits.second[task].load() == 1);
    for (std::size_t thread = 0; thread != threads; ++thread) expect_true(visits.third[thread].load() == 1);
    expect_true(visits.refused.load() == threads - 1);

    std::atomic<std::size_t> plain_visits {0};
    pool.for_threads([&](std::size_t) noexcept { plain_visits.fetch_add(1, std::memory_order_relaxed); });
    expect_true(plain_visits.load() == threads);
}

/** Verifies guard timing: dispatch at construction, and the inclusive caller's slice at join. */
template <typename make_pool_type_ = make_pool_t>
static void test_broadcast_join_dispatches_at_construction() noexcept {

    auto maker = make_pool_type_ {};

    // On exclusive pools the work starts at construction, before `join`:
    {
        auto pool = maker.construct();
        if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        auto broadcast = pool.for_threads(count_visits);
        expect_true(spin_until([&] { return broadcast.is_complete(); })); // ? Wait without joining
        expect_equal(visited_count.load(std::memory_order_relaxed), pool.threads_count());
        broadcast.join();
    }

    // On inclusive pools the workers start at construction too, and the caller's slice at `join`:
    {
        auto pool = maker.construct();
        if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        auto broadcast = pool.for_threads(count_visits);
        // ? Every worker, no join
        expect_true(
            spin_until([&] { return visited_count.load(std::memory_order_relaxed) == pool.threads_count() - 1; }));
        expect_true(!broadcast.is_complete());
        broadcast.join();
        expect_true(broadcast.is_complete());
        expect_equal(visited_count.load(std::memory_order_relaxed), pool.threads_count());
    }
}

/** Dropping the guard without an explicit @c join must still join in the destructor. */
template <typename make_pool_type_ = make_pool_t>
static void test_broadcast_join_destructor_joins() noexcept {

    auto maker = make_pool_type_ {};
    for (fu::caller_exclusivity_t const exclusivity : {fu::caller_exclusive_k, fu::caller_inclusive_k}) {
        auto pool = maker.construct();
        if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), exclusivity)))) return;

        std::atomic<std::size_t> visited_count {0};
        auto count_visits = [&](std::size_t) noexcept { visited_count.fetch_add(1, std::memory_order_relaxed); };
        { auto broadcast = pool.for_threads(count_visits); } // ! The destructor is the only join here
        expect_equal(visited_count.load(std::memory_order_relaxed), pool.threads_count());
    }
}

/** Covers the caller-as-contributor protocol on inclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_inclusive_join_runs_the_caller_slice() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;

    std::vector<std::atomic<bool>> visited(pool.threads_count());
    auto mark_visited = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect_true(!pool.is_complete(generation)); // ? Impossible before the caller's slice
    pool.unsafe_join(generation);               // ? Runs the caller's slice, then waits
    expect_true(pool.is_complete(generation));
    for (std::size_t i = 0; i < pool.threads_count(); ++i) expect_true(visited[i]);
    pool.unsafe_join(generation); // ? Idempotent: double-join must be a no-op
}

/** Degenerate single-thread inclusive pool: the caller is the only contributor. */
static void test_generation_on_a_lone_caller() noexcept {
    fu::flat_pool_t pool;
    if (!expect_true(fu::succeeded(pool.spawn(1)))) return; // ? Default is caller-inclusive: zero workers

    std::atomic<bool> visited {false};
    auto mark_visited = [&](std::size_t) noexcept { visited.store(true, std::memory_order_relaxed); };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect_true(!pool.is_complete(generation)); // ? Nothing can complete before the caller's slice
    pool.unsafe_join(generation);               // ? The caller both runs and signals completion
    expect_true(pool.is_complete(generation));
    expect_true(visited.load(std::memory_order_relaxed));
}

/** The exclusive mirror: one lone worker, while the caller only polls and never contributes. */
static void test_generation_on_a_lone_worker() noexcept {
    fu::flat_pool_t pool;
    if (!expect_true(fu::succeeded(pool.spawn(1, fu::caller_exclusive_k)))) return;

    std::atomic<bool> visited {false};
    auto mark_visited = [&](std::size_t) noexcept { visited.store(true, std::memory_order_relaxed); };

    auto generation = pool.unsafe_for_threads(mark_visited);
    expect_true(spin_until([&] { return pool.is_complete(generation); })); // ? The lone worker finishes alone
    pool.unsafe_join(generation);
    expect_true(visited.load(std::memory_order_relaxed));
}

/** Hammers the dispatch/join race window with tight iterations on exclusive pools. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_survives_back_to_back_dispatches() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    auto polled_pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;
    if (!expect_true(fu::succeeded(polled_pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

    std::atomic<std::size_t> counter {0};
    auto count_up = [&](std::size_t) noexcept { counter.fetch_add(1, std::memory_order_relaxed); };

    // A second in-flight pool is polled between iterations to stress `is_complete`
    auto polled_generation = polled_pool.unsafe_for_threads(count_up);

    std::size_t const iterations = scaled_iterations(10000);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        auto generation = pool.unsafe_for_threads(count_up);
        (void)polled_pool.is_complete(polled_generation);
        pool.unsafe_join(generation);
        expect_true(pool.is_complete(generation));
    }
    polled_pool.unsafe_join(polled_generation);

    std::size_t const expected = pool.threads_count() * iterations + polled_pool.threads_count();
    expect_equal(counter.load(std::memory_order_relaxed), expected);
}

/** Overlaps an inclusive and an exclusive pool from one caller, through the raw token API. */
template <typename make_pool_type_ = make_pool_t>
static void test_mixed_exclusivity_overlaps_through_tokens() noexcept {

    auto maker = make_pool_type_ {};
    auto first_pool = maker.construct();
    auto second_pool = maker.construct();
    if (!expect_true(fu::succeeded(first_pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;
    if (!expect_true(fu::succeeded(second_pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

    std::size_t const first_size = first_pool.threads_count();
    std::size_t const second_size = second_pool.threads_count();
    std::size_t const total_size = first_size + second_size;
    std::vector<std::atomic<bool>> visited(total_size);

    // Externally defined lambdas with a clearly long lifetime:
    auto do_second = [&](std::size_t const thread_index) noexcept {
        visited[first_size + thread_index].store(true, std::memory_order_relaxed);
    };
    auto do_first = [&](std::size_t const thread_index) noexcept {
        visited[thread_index].store(true, std::memory_order_relaxed);
    };

    // Repeat the same logic a few times and check for correctness:
    for (std::size_t iteration = 0; iteration < 3; ++iteration) {
        auto second_generation = second_pool.unsafe_for_threads(do_second);
        auto first_generation = first_pool.unsafe_for_threads(do_first);
        first_pool.unsafe_join(first_generation); // ? Contributes the caller's slice: inclusive pool
        second_pool.unsafe_join(second_generation);

        for (std::size_t i = 0; i < total_size; ++i) expect_true(visited[i]);
    }
}

/** The same overlap through the guard API: inline lambdas re-packaged into returned objects. */
template <typename make_pool_type_ = make_pool_t>
static void test_mixed_exclusivity_overlaps_through_guards() noexcept {

    auto maker = make_pool_type_ {};
    auto first_pool = maker.construct();
    auto second_pool = maker.construct();
    if (!expect_true(fu::succeeded(first_pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;
    if (!expect_true(fu::succeeded(second_pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

    std::size_t const first_size = first_pool.threads_count();
    std::size_t const second_size = second_pool.threads_count();
    std::size_t const total_size = first_size + second_size;
    std::vector<std::atomic<bool>> visited(total_size);

    auto join_second = second_pool.for_threads([&](std::size_t const thread_index) noexcept {
        visited[first_size + thread_index].store(true, std::memory_order_relaxed);
    });
    first_pool.for_threads(
        [&](std::size_t const thread_index) noexcept { visited[thread_index].store(true, std::memory_order_relaxed); });
    join_second.join();

    for (std::size_t i = 0; i < total_size; ++i) expect_true(visited[i]);
}

/** Make sure that @c for_n is called from each thread. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_across_uncomfortable_sizes() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::size_t const max_input_size = pool.threads_count() * 3; // Arbitrary size, larger than the number of threads
    for (std::size_t input_size = 0; input_size <= max_input_size; ++input_size) {
        std::atomic<bool> out_of_bounds(false);
        std::atomic<std::size_t> executed(0);
        auto probe = [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
            if (task >= input_size) out_of_bounds.store(true, std::memory_order_relaxed);
            executed.fetch_add(1, std::memory_order_relaxed);
        };

        // Both schedulers must cover the awkward sizes exactly; a drop is as wrong as a duplicate.
        pool.for_n(input_size, probe);
        expect_equal(executed.load(std::memory_order_relaxed), input_size);

        executed.store(0, std::memory_order_relaxed);
        pool.for_n_dynamic(input_size, probe);
        expect_equal(executed.load(std::memory_order_relaxed), input_size);
        expect_true(!out_of_bounds.load(std::memory_order_relaxed));
    }
}

/**
 *  @brief One @c for_slices dispatch of @p n tasks must partition [0, n) into non-empty slices.
 *
 *  A slice callback receives a @c tasks_range_t, so a broken split shows up as an index covered
 *  twice, never, or past the range. Every thread is dispatched exactly once, an idle one with an
 *  empty range, and no dispatch may produce more ranges than there are threads.
 */
template <typename pool_type_>
static void check_for_slices_cover(pool_type_ &pool, std::size_t const n) noexcept {
    std::vector<std::atomic<unsigned>> executions(n);
    std::atomic<std::size_t> slices_count {0};
    std::atomic<bool> out_of_bounds {false};

    pool.for_slices(n, [&](fu::tasks_range_t range, fu::thread_in_domain_t) noexcept {
        if (range.first + range.count > n) {
            out_of_bounds.store(true, std::memory_order_relaxed);
            return;
        }
        slices_count.fetch_add(1, std::memory_order_relaxed);
        for (std::size_t const task : range) executions[task].fetch_add(1, std::memory_order_relaxed);
    });

    // Every thread is dispatched exactly once, whether or not it drew any tasks.
    expect_equal(slices_count.load(std::memory_order_relaxed), pool.threads_count());
    expect_true(!out_of_bounds.load(std::memory_order_relaxed));
    expect_true(slices_count.load(std::memory_order_relaxed) <= pool.threads_count());
    for (std::size_t i = 0; i < n; ++i) expect_equal(executions[i].load(std::memory_order_relaxed), 1u);
}

/** Sweeps @c for_slices through the awkward sizes around the thread count and one large run. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_slices_partitions_the_range() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::size_t const threads = pool.threads_count();
    std::size_t const sizes[] = {0, 1, threads - 1, threads, threads + 1, 3 * threads, default_parallel_tasks_k};
    for (std::size_t const n : sizes) check_for_slices_cover(pool, n);
}

/** A joined token must stay complete even while a newer fork is in flight. */
template <typename make_pool_type_ = make_pool_t>
static void test_generation_stale_token_stays_complete() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_exclusive_k)))) return;

    std::atomic<std::size_t> counter {0};
    auto count_up = [&](std::size_t) noexcept { counter.fetch_add(1, std::memory_order_relaxed); };

    auto stale = pool.unsafe_for_threads(count_up);
    pool.unsafe_join(stale);
    expect_true(pool.is_complete(stale));

    auto fresh = pool.unsafe_for_threads(count_up);
    expect_true(stale != fresh);
    expect_true(pool.is_complete(stale)); // ? A joined token must not flip back while a new fork runs
    pool.unsafe_join(fresh);
    expect_true(pool.is_complete(fresh));
    expect_equal(counter.load(std::memory_order_relaxed), 2 * pool.threads_count());
}

/**
 *  @brief Hammers the spawn/terminate lifecycle; every re-spawned crew must dispatch exactly-once.
 *
 *  Worker ids are published by the workers themselves and reset on @c terminate, so lifecycle churn
 *  is where a stale id or an unjoined thread would surface - as a lost task or a crash on re-spawn.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_spawn_terminate_churn_loses_nothing() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();

    std::size_t const rounds = scaled_iterations(16);
    std::atomic<std::size_t> executed {0};
    for (std::size_t round = 0; round != rounds; ++round) {
        if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;
        executed.store(0, std::memory_order_relaxed);
        std::size_t const n = pool.threads_count() + round % 3; // ? Vary the remainder across rounds
        pool.for_n(
            n, [&](std::size_t, fu::thread_in_domain_t) noexcept { executed.fetch_add(1, std::memory_order_relaxed); });
        expect_equal(executed.load(std::memory_order_relaxed), n);
        pool.terminate();
    }
}

/** Two caller threads drive two independent pools at once; no cross-pool state may bleed. */
static void test_two_callers_drive_two_pools_apart() noexcept {
    constexpr std::size_t tasks_k = 4096;
    std::atomic<bool> first_ok {true}, second_ok {true};

    auto drive = [&](std::atomic<bool> &ok) noexcept {
        fu::flat_pool_t pool;
        if (fu::failed(pool.spawn(fu::allowed_cores_count()))) {
            ok.store(false, std::memory_order_relaxed);
            return;
        }
        std::atomic<std::size_t> counter {0};
        for (std::size_t round = 0; round != 8; ++round) {
            counter.store(0, std::memory_order_relaxed);
            pool.for_n(tasks_k, [&](std::size_t, fu::thread_in_domain_t) noexcept {
                counter.fetch_add(1, std::memory_order_relaxed);
            });
            if (counter.load(std::memory_order_relaxed) != tasks_k) {
                ok.store(false, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::thread first([&] { drive(first_ok); });
    std::thread second([&] { drive(second_ok); });
    first.join();
    second.join();
    expect_true(first_ok.load(std::memory_order_relaxed));
    expect_true(second_ok.load(std::memory_order_relaxed));
}

/** Convenience structure to ensure we output match locations to independent cache lines. */
struct alignas(fu::default_alignment_k) aligned_visit_t {
    std::size_t task = 0;
    auto operator<=>(aligned_visit_t const &) const noexcept = default;
    bool operator==(std::size_t other_index) const noexcept { return task == other_index; }
};

/** Sorts the visit records in place and answers whether they cover [0, size) exactly once. */
static bool contains_iota(std::vector<aligned_visit_t> &visited) noexcept {
    std::sort(visited.begin(), visited.end());
    std::size_t covered = 0;
    while (covered != visited.size() && visited[covered] == covered) ++covered;
    return covered == visited.size();
}

/** Make sure that @c for_n is called the right number of times with the right task indices. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_visits_each_task_once_and_again() noexcept {

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0,
    // `default_parallel_tasks_k`).
    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0,
    // `default_parallel_tasks_k`).
    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));

    // Make sure `for_n` is being executed on different threads.
    std::vector<aligned_visit_t> visited_threads(pool.threads_count());
    constexpr std::size_t invalid_task = (std::numeric_limits<std::size_t>::max)();
    for (auto &visit : visited_threads) visit.task = invalid_task;
    pool.for_n(
        default_parallel_tasks_k, // ? Could have been an arbitrary number `>= pool.threads_count()`
        [&](std::size_t const, fu::thread_in_domain_t at) noexcept { visited_threads[at.thread].task = at.thread; });

    expect_true(contains_iota(visited_threads));
}

/** Ensures @c for_n_dynamic runs the right number of times with the right task indices. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_dynamic_covers_uneven_work() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0,
    // `default_parallel_tasks_k`).
    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));
}

/**
 *  @brief Stalls one thread and checks its neighbours drain the slice it can't reach.
 *
 *  @c for_n_dynamic reserves a contiguous slice per thread and lets idle threads help drain the
 *  rest. Correctness must not depend on any thread keeping up: with one thread crawling, every task
 *  still runs, and runs @b once. The stalled thread must also end up executing fewer tasks than its
 *  own slice held - otherwise nobody stole, and the test is silently proving nothing.
 */
template <typename make_pool_type_ = make_pool_t>
static case_result_t test_for_n_dynamic_steals_from_a_stalled_thread() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return {};

    std::size_t const threads = pool.threads_count();
    if (threads < 2) return case_result_t::skipped("needs 2+ threads"); // ? Nobody to steal from, and nobody to steal

    constexpr std::size_t tasks_k = 8192;

    std::atomic<std::size_t> total {0};
    std::atomic<std::size_t> thread_0_runs {0};
    std::vector<std::atomic<unsigned>> executions(tasks_k);
    for (auto &e : executions) e.store(0, std::memory_order_relaxed);

    pool.for_n_dynamic(tasks_k, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // Thread 0 crawls: a spin, not a sleep, so the pool's own yields don't mask the stall.
        if (at.thread == 0) {
            volatile std::size_t sink = 0;
            for (std::size_t i = 0; i < 200000; ++i) sink = sink + i;
            thread_0_runs.fetch_add(1, std::memory_order_relaxed);
        }
        executions[task].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(1, std::memory_order_relaxed);
    });

    expect_equal(total.load(), tasks_k);                                              // ! Some task ran twice, or never
    for (std::size_t i = 0; i < tasks_k; ++i) expect_equal(executions[i].load(), 1u); // ! Every task exactly once

    // Thread 0's own slice - had nobody helped, it would have run all of it, plus a static prong.
    std::size_t const dynamic_tasks = tasks_k > threads ? tasks_k - threads : 0;
    fu::indexed_split<std::size_t> const split(dynamic_tasks, threads);
    std::size_t const own_slice = split[0].count;
    expect_true(thread_0_runs.load() < own_slice); // ! Nobody stole, so this test proves nothing
    return {};
}

/**
 *  @brief A @c for_n_dynamic the pool refuses leaves the running dispatch's cursors alone.
 *
 *  The cursors live in the pool, so a refused call that rewound them would hand the running
 *  dispatch's tasks out again. Both refusals are checked: a call nested in a task of the running
 *  dispatch, and a second call while the caller still holds the first unjoined.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_dynamic_refusals_leave_the_cursors() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(), fu::caller_inclusive_k)))) return;

    constexpr std::size_t tasks_k = 1024;
    std::vector<std::atomic<unsigned>> executions(tasks_k);
    for (auto &e : executions) e.store(0, std::memory_order_relaxed);
    auto run_once = [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        executions[task].fetch_add(1, std::memory_order_relaxed);
    };

    std::atomic<std::size_t> admitted {0};
    pool.for_n_dynamic(tasks_k, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        run_once(task, at);
        if (pool.for_n_dynamic(1, run_once).generation() != 0) admitted.fetch_add(1, std::memory_order_relaxed);
    });
    expect_equal(admitted.load(), 0u);                                                // ! Admitted inside itself
    for (std::size_t i = 0; i < tasks_k; ++i) expect_equal(executions[i].load(), 1u); // ! A refusal rewound

    for (auto &e : executions) e.store(0, std::memory_order_relaxed);
    {
        auto held = pool.for_n_dynamic(tasks_k, run_once);
        expect_true(pool.for_n_dynamic(tasks_k, run_once).generation() == 0); // ! Admitted beside a held dispatch
    }
    for (std::size_t i = 0; i < tasks_k; ++i) expect_equal(executions[i].load(), 1u); // ! A refusal rewound
}

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN
/**
 *  @brief One dynamic dispatch of @p n tasks must run exactly-once; @p crawl stalls a whole domain.
 *
 *  With domain 0 crawling, the other domains drain their own slices and must reach across the
 *  interconnect for more, so the steal path is exercised rather than silently idle. The owner of a
 *  task index mirrors the invoker's own split: domain @c d owns `split[d]`.
 *
 *  @note The crawl is @b per @b domain, not per thread - stalling one thread may not force it.
 *
 *  With 64 threads to a domain, one crawler is 1/64th of its capacity, its neighbours absorb the
 *  slice, and no steal ever needs to cross - so the assertion below passed on small CI runners and
 *  was a coin-flip on real hardware.
 */
template <typename pool_type_>
static void check_dynamic_regime_covers(pool_type_ &pool, std::size_t const n, bool const crawl) noexcept {
    std::size_t const domains = pool.compute_domains_count();

    std::atomic<std::size_t> total {0};
    std::atomic<std::size_t> cross_domain_runs {0};
    std::vector<std::atomic<unsigned>> executions(n);
    for (auto &e : executions) e.store(0, std::memory_order_relaxed);

    pool.for_n_dynamic(n, [&](std::size_t const task, fu::thread_in_domain_t at) noexcept {
        // ? A spin, not a sleep, so the pool's yields don't mask it. Keyed on the executing
        // thread's domain, so a stealer from elsewhere runs the stolen task at full speed.
        if (crawl && at.compute_domain == 0) {
            volatile std::size_t sink = 0;
            for (std::size_t i = 0; i < 200000; ++i) sink = sink + i;
        }
        std::size_t const owner_domain = fu::indexed_split<std::size_t>(n, domains).index_of(task);
        if (owner_domain != static_cast<std::size_t>(at.compute_domain))
            cross_domain_runs.fetch_add(1, std::memory_order_relaxed);
        executions[task].fetch_add(1, std::memory_order_relaxed);
        total.fetch_add(1, std::memory_order_relaxed);
    });

    expect_equal(total.load(), n);                                              // ! Some task ran twice, or never
    for (std::size_t i = 0; i < n; ++i) expect_equal(executions[i].load(), 1u); // ! Every task exactly once
    if (crawl) expect_true(cross_domain_runs.load() > 0); // ! No steal crossed the interconnect; proves nothing
}

/**
 *  @brief Exhausts @c for_n_dynamic across task-count regimes on a real multi-domain box.
 *
 *  Every boundary the two-level slicing exposes gets a regime - fewer tasks than threads, exactly
 *  as many, one more, and a large skewed run with a crawling thread. The index type here is
 *  @c std::size_t; small-type wrap-around regimes live in the flat pool's @c fu8 and @c fu16 suite.
 */
template <typename make_pool_type_ = make_pool_t>
static case_result_t test_for_n_dynamic_covers_every_regime_across_domains() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return {};

    std::size_t const threads = pool.threads_count();
    std::size_t const domains = pool.compute_domains_count();
    // ? The cross-domain walk is dead code otherwise
    if (domains < 2) return case_result_t::skipped("needs 2+ compute domains");
    if (threads < 4) return case_result_t::skipped("needs 4+ threads");

    std::size_t const regimes[] = {1, threads / 2, threads, threads + 1, 2 * threads + 3, 8192};
    for (std::size_t const n : regimes) {
        bool const crawl = n == 8192; // ? Only the large run needs forced imbalance
        check_dynamic_regime_covers(pool, n, crawl);
    }
    return {};
}
#endif // FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/** Stress-tests the implementation by oversubscribing the number of threads. */
template <typename make_pool_type_ = make_pool_t>
static void test_for_n_dynamic_oversubscribed() noexcept {
    constexpr std::size_t oversubscription = 3;

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope(oversubscription))))) return;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);
    thread_local volatile std::size_t some_local_work = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // Perform some weird amount of work, that is not very different between consecutive tasks.
        for (std::size_t i = 0; i != task % oversubscription; ++i) some_local_work = some_local_work + i * i;

        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    // Make sure that all task indices are unique and form the full range of [0,
    // `default_parallel_tasks_k`).
    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));
}

/**
 *  @brief Parks the workers between batches; every wake must still dispatch exactly-once.
 *
 *  @c nap parks them on a monitor and @c sleep in the kernel; the next dispatch must wake them from
 *  either, whichever came last. Losing a worker to a missed wake-up shows up here as a hung join,
 *  and a double-dispatched task as a broken iota.
 */
template <typename make_pool_type_ = make_pool_t>
static void test_sleep_then_wake_loses_nothing() noexcept {
    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter {0};
    for (std::size_t batch = 0; batch != 4; ++batch) {
        if (batch == 0) pool.nap(), pool.nap();
        if (batch == 1) pool.sleep(100), pool.sleep();
        if (batch == 2) pool.nap(), pool.sleep();
        if (batch == 3) pool.sleep(), pool.nap();
        counter.store(0, std::memory_order_relaxed);
        pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
            std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
            visited[count_populated].task = task;
        });
        expect_equal(counter.load(std::memory_order_relaxed), default_parallel_tasks_k);
        expect_true(contains_iota(visited));
    }
}

/** Combines static and dynamic loads over the same pool, with and without resetting. */
template <bool should_restart_, typename make_pool_type_ = make_pool_t>
static void test_static_and_dynamic_dispatch_share_a_pool() noexcept {

    auto maker = make_pool_type_ {};
    auto pool = maker.construct();
    if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;

    std::vector<aligned_visit_t> visited(default_parallel_tasks_k);
    std::atomic<std::size_t> counter(0);

    pool.for_n(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));

    // Make sure that the pool can be reset and reused
    if (should_restart_) {
        pool.terminate();
        if (!expect_true(fu::succeeded(pool.spawn(maker.scope())))) return;
    }

    // Make sure repeated calls to `for_n` work
    counter = 0;
    pool.for_n_dynamic(default_parallel_tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });

    expect_equal(counter.load(), default_parallel_tasks_k);
    expect_true(contains_iota(visited));
}

/**
 *  @brief Alternates static & dynamic dispatch over a small-index pool for many generations.
 *
 *  The epoch is the same width as the pool's index, so a @c fu8 pool wraps it every 128
 *  dispatch/join cycles and a @c fu16 every 32768. Enough @p cycles drives the counter through
 *  several wraps, proving the token parity and completion comparisons survive the overflow; the
 *  run's scale shrinks them, never below one wrap.
 */
template <typename pool_type_>
static case_result_t test_mixed_dispatch_survives_epoch_wraps(std::size_t const threads_count,
                                                              std::size_t const parallel_tasks_count,
                                                              std::size_t const cycles) noexcept {
    using pool_t = pool_type_;
    using index_t = typename pool_t::index_t;

    pool_t pool;
    if (!expect_true(fu::succeeded(pool.spawn(static_cast<index_t>(threads_count))))) return {};

    std::atomic<std::size_t> counter(0);
    std::vector<aligned_visit_t> visited(parallel_tasks_count);

    // ? Relax the memory order - result order doesn't matter, we'll sort 'em later
    auto log_visit = [&](index_t const task, typename pool_t::thread_in_domain_t) noexcept {
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    };
    auto dispatch_and_verify = [&](auto dispatch) noexcept {
        counter = 0;
        dispatch();
        expect_equal(counter.load(), parallel_tasks_count);
        expect_true(contains_iota(visited));
    };

    index_t const n = static_cast<index_t>(parallel_tasks_count);
    // Two dispatches a cycle, so a quarter of the index range plus one wraps the epoch once
    std::size_t const one_wrap = (static_cast<std::size_t>((std::numeric_limits<index_t>::max)()) + 1) / 4 + 1;
    std::size_t const scaled_cycles = cycles > 1 ? fu::max_of(scaled_iterations(cycles), one_wrap) : cycles;
    for (std::size_t cycle = 0; cycle != scaled_cycles; ++cycle) {
        dispatch_and_verify([&] { pool.for_n(n, log_visit); });
        dispatch_and_verify([&] { pool.for_n_dynamic(n, log_visit); });
    }
    return {};
}

/** @c replicated_array is one uninitialized per-domain buffer; the caller fills and reads its
 *  slices. */
static case_result_t test_replicated_array_gives_every_domain_a_copy() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest()))
        return case_result_t::skipped("fu::machine_topology_t::harvest is refused on this machine");

    std::size_t const n = 4096;
    fu::replicated_array<std::uint32_t> replicas;
    if (!expect_true(fu::succeeded(replicas.resize_uninitialized(topology, n)))) return {};
    expect_equal(replicas.size(), n);
    expect_equal(replicas.memory_domains_count(), topology.memory_domains_count());

    // Fill every replica with a domain-dependent pattern, so replicas that aliased would be caught.
    for (std::size_t domain = 0; domain < replicas.memory_domains_count(); ++domain) {
        std::span<std::uint32_t> const replica =
            replicas.on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
        if (!expect_equal(replica.size(), n)) return {};
        for (std::size_t i = 0; i < n; ++i) replica[i] = static_cast<std::uint32_t>(domain * n + i);
    }
    for (std::size_t domain = 0; domain < replicas.memory_domains_count(); ++domain)
        for (std::size_t i = 0; i < n; ++i)
            expect_equal(replicas.at(static_cast<fu::memory_domain_index_t>(domain), i), domain * n + i);

    // A single-element replica is the smallest shape the stride math must survive.
    fu::replicated_array<std::uint32_t> tiny;
    if (!expect_true(fu::succeeded(tiny.resize_uninitialized(topology, 1)))) return {};
    expect_equal(tiny.size(), 1u);
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        tiny.at(static_cast<fu::memory_domain_index_t>(domain), 0) = static_cast<std::uint32_t>(domain);
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        expect_equal(tiny.at(static_cast<fu::memory_domain_index_t>(domain), 0), domain);
    return {};
}

/** @c sharded_array stores each element once; @c location_of and @c logical_index_of round-trip the
 *  segment map. */
static case_result_t test_sharded_array_round_trips_its_segments() noexcept {
    fu::machine_topology_t topology;
    if (fu::failed(topology.harvest()))
        return case_result_t::skipped("fu::machine_topology_t::harvest is refused on this machine");

    std::size_t const n = 4096;
    fu::sharded_array<std::uint32_t> shards;
    if (!expect_true(fu::succeeded(shards.resize_uninitialized(topology, n)))) return {};

    // Each element lives exactly once: the shard lengths sum to the logical length.
    std::size_t footprint = 0;
    for (std::size_t domain = 0; domain < shards.memory_domains_count(); ++domain)
        footprint += shards.length_on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
    expect_equal(shards.size(), n);
    expect_equal(footprint, n);

    // Write each element to its own logical index, then read it back through `location_of`.
    for (std::size_t domain = 0; domain < shards.memory_domains_count(); ++domain) {
        auto const memory_domain = static_cast<fu::memory_domain_index_t>(domain);
        for (std::size_t local = 0; local < shards.length_on_memory_domain(memory_domain); ++local)
            shards.at(memory_domain, local) = static_cast<std::uint32_t>(shards.logical_index_of(memory_domain, local));
    }
    for (std::size_t i = 0; i < n; ++i) {
        fu::sharded_array<std::uint32_t>::location_t const home = shards.location_of(i);
        expect_equal(shards.at(home.memory_domain, home.local_index), i);
    }

    // A logical length shorter than the domain count leaves trailing shards empty, never negative.
    fu::sharded_array<std::uint32_t> tiny;
    if (!expect_true(fu::succeeded(tiny.resize_uninitialized(topology, 1)))) return {};
    expect_equal(tiny.segment(), 1u);
    std::size_t tiny_footprint = 0;
    for (std::size_t domain = 0; domain < tiny.memory_domains_count(); ++domain)
        tiny_footprint += tiny.length_on_memory_domain(static_cast<fu::memory_domain_index_t>(domain));
    expect_equal(tiny_footprint, 1u);
    fu::sharded_array<std::uint32_t>::location_t const home = tiny.location_of(0);
    tiny.at(home.memory_domain, home.local_index) = 42u;
    expect_equal(tiny.at(home.memory_domain, home.local_index), 42u);
    return {};
}

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

/** One distributed pool of exactly @p threads workers must dispatch every task exactly once. */
static void check_spawn_shape_dispatches(std::size_t const threads) noexcept {
    constexpr std::size_t tasks_k = 4096;
    std::vector<aligned_visit_t> visited(tasks_k);
    std::atomic<std::size_t> counter {0};

    fu::distributed_pool_t pool("forkunion");
    if (!expect_true(fu::succeeded(pool.spawn(machine_topology(), threads)))) return;
    expect_equal(pool.threads_count(), threads); // ! A shape was silently resized

    // A pool no larger than the machine never doubles its workers up on any domain's cores.
    for (std::size_t colocation = 0; colocation != pool.compute_domains_count(); ++colocation) {
        fu::compute_domain_t const &domain = machine_topology().compute_domain_at(
            static_cast<fu::compute_domain_index_t>(pool.compute_domain_index_of(colocation)));
        if (threads <= machine_topology().logical_cores_count())
            expect_true(pool.threads_count(colocation) <= domain.core_ids.size());
    }

    pool.for_n_dynamic(tasks_k, [&](std::size_t const task, fu::thread_in_domain_t) noexcept {
        std::size_t const count_populated = counter.fetch_add(1, std::memory_order_relaxed);
        visited[count_populated].task = task;
    });
    expect_equal(counter.load(std::memory_order_relaxed), tasks_k);
    expect_true(contains_iota(visited));
    pool.terminate();
}

/**
 *  @brief Spawns the distributed pool at awkward worker counts and checks exactly-once dispatch.
 *
 *  A pool sized to the machine never exercises the remainder paths of the per-domain split; a
 *  worker count that is prime to the domain count, short of the cores, or past them does.
 */
static void test_awkward_spawn_shapes_dispatch_once() noexcept {
    std::size_t const cores = machine_topology().logical_cores_count();
    std::size_t const shapes[] = {1, 2, cores > 1 ? cores - 1 : 1, cores + 3};
    for (std::size_t const threads : shapes) check_spawn_shape_dispatches(threads);
}
#endif // FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN && FORKUNION_OS_LINUX_ && FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY

/**
 *  @brief A pool sizes itself from the cores we were given, and leaves the caller its own mask.
 *
 *  The process is narrowed here the way @c taskset or a cgroup @c cpuset would narrow it. A
 *  topology harvested from the machine rather than from the mask would report every core, the pool
 *  would oversubscribe them, and a "restore" that widens to the machine would leave the caller
 *  running on cores this process was never granted.
 */
static case_result_t test_caller_affinity_is_preserved() noexcept {
    fu::core_mask_t original;
    if (fu::failed(fu::capture_thread_cores(original)))
        return case_result_t::skipped("fu::capture_thread_cores is refused on this machine");
    if (original.count() < 2) return case_result_t::skipped("needs 2+ allowed cores"); // ? Too narrow to narrow further

    // Narrow the caller the way `taskset` would: keep the two lowest allowed cores.
    fu::core_mask_t narrowed;
    if (!expect_true(fu::succeeded(narrowed.resize()))) return {};
    std::size_t kept = 0;
    for (std::size_t cpu = 0; cpu < original.capacity() && kept < 2; ++cpu)
        if (original.contains(static_cast<fu::core_id_t>(cpu))) narrowed.add(static_cast<fu::core_id_t>(cpu)), ++kept;

    bool succeeded = fu::succeeded(fu::restore_thread_cores(narrowed)); // ? Applies `narrowed`
    if (succeeded) {
        fu::machine_topology_t topology;
        succeeded = fu::succeeded(topology.harvest()) && topology.logical_cores_count() == 2;

        if (succeeded) {
            fu::distributed_pool_t pool;
            succeeded = fu::succeeded(pool.spawn(topology, 2));
            if (succeeded) succeeded = pool.all_threads_pinned();
            // The caller joins as slot 0 but keeps its own two cores while the pool lives
            fu::core_mask_t during;
            succeeded = succeeded && fu::succeeded(fu::capture_thread_cores(during)) && during.count() == 2;
            pool.terminate();
        }

        // The caller is still on the two cores it narrowed itself to, not the machine's.
        fu::core_mask_t afterwards;
        succeeded = succeeded && fu::succeeded(fu::capture_thread_cores(afterwards)) && afterwards.count() == 2;
    }

    (void)fu::restore_thread_cores(original); // ? Leave the process as we found it
    expect_true(succeeded);
    return {};
}
#endif // FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN && FORKUNION_OS_LINUX_ && FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY

#if defined(__cpp_lib_atomic_ref) && defined(__cpp_lib_bit_cast)

/** @p pattern cut to the word's width; casting the literal itself warns under MSVC. */
template <typename word_type_>
static constexpr word_type_ make_pattern(int pattern) noexcept {
    return word_type_(pattern);
}

/** The standard operations a reference spells by hand: every return value is the word before. The
 *  bit patterns truncate with the word, so one set serves every width. */
template <template <typename> class atomic_ref_, typename word_type_>
static void check_atomic_ref_words() noexcept {
    word_type_ word = 5;
    atomic_ref_<word_type_> reference(word);
    expect_equal(reference.exchange(word_type_(9), std::memory_order_acq_rel), word_type_(5));
    expect_equal(reference.fetch_add(word_type_(3), std::memory_order_acquire), word_type_(9));
    expect_equal(reference.fetch_sub(word_type_(2), std::memory_order_relaxed), word_type_(12));
    expect_equal(reference.load(std::memory_order_acquire), word_type_(10));
    word_type_ expected = 10;
    expect_true(reference.compare_exchange_strong(expected, word_type_(11), std::memory_order_acq_rel,
                                                  std::memory_order_acquire));
    expected = 3;
    expect_true(!reference.compare_exchange_weak(expected, word_type_(12), std::memory_order_acquire,
                                                 std::memory_order_relaxed));
    expect_equal(expected, word_type_(11)); // ? The failed compare reports what it saw
    reference.store(word_type_(6), std::memory_order_release);
    expect_equal(word, word_type_(6));

    word_type_ bits = make_pattern<word_type_>(0xF0F0);
    atomic_ref_<word_type_> bits_reference(bits);
    expect_equal(bits_reference.fetch_and(make_pattern<word_type_>(0xFF00), std::memory_order_acq_rel),
                 make_pattern<word_type_>(0xF0F0));
    expect_equal(bits_reference.fetch_or(word_type_(0x1), std::memory_order_acq_rel), make_pattern<word_type_>(0xF000));
    expect_equal(bits_reference.fetch_xor(word_type_(0xFF), std::memory_order_acq_rel),
                 make_pattern<word_type_>(0xF001));
    expect_equal(bits, make_pattern<word_type_>(0xF0FE));

    bool flag = false;
    atomic_ref_<bool> flag_reference(flag);
    expect_equal(flag_reference.exchange(true, std::memory_order_acquire), false);
    flag_reference.store(false, std::memory_order_release);
    expect_equal(flag, false);
}

/** The operations past the standard - no-return forms, @c fetch_max and @c fetch_min, the
 *  conditional adds - each refusing exactly at its bound, unsigned and signed alike. */
template <template <typename> class atomic_ref_, typename word_type_>
static void check_atomic_ref_extensions() noexcept {
    word_type_ word = 11;
    atomic_ref_<word_type_> reference(word);
    expect_equal(reference.fetch_max(word_type_(7), std::memory_order_acq_rel),
                 word_type_(11)); // ? Loses: writes nothing
    expect_equal(reference.fetch_max(word_type_(12), std::memory_order_acq_rel), word_type_(11)); // ? Wins
    expect_equal(reference.fetch_min(word_type_(3), std::memory_order_relaxed), word_type_(12));
    reference.store_max(word_type_(9), std::memory_order_release);
    reference.store_min(word_type_(3), std::memory_order_relaxed);
    expect_equal(word, word_type_(3)); // ? 3 rose to 9, then fell back to 3
    reference.store_add(word_type_(4), std::memory_order_release);
    reference.store_sub(word_type_(2), std::memory_order_relaxed);
    expect_equal(word, word_type_(5));
    expect_equal(reference.fetch_add_if_at_most(word_type_(3), word_type_(8), std::memory_order_acq_rel),
                 word_type_(5)); // 5 + 3 <= 8: adds
    expect_equal(reference.fetch_add_if_at_most(word_type_(1), word_type_(8), std::memory_order_acq_rel),
                 word_type_(8)); // 8 + 1 > 8: refuses
    expect_equal(reference.fetch_sub_if_at_least(word_type_(8), word_type_(0), std::memory_order_acquire),
                 word_type_(8)); // 8 - 8 >= 0: subtracts
    expect_equal(reference.fetch_sub_if_at_least(word_type_(1), word_type_(0), std::memory_order_acquire),
                 word_type_(0)); // 0 - 1 < 0: refuses
    expect_equal(reference.fetch_add_if_at_most(word_type_(9), word_type_(8), std::memory_order_acq_rel),
                 word_type_(0)); // ? An operand past the limit
    expect_equal(reference.fetch_sub_if_at_least(word_type_(1), word_type_(-1), std::memory_order_acquire),
                 word_type_(0)); // ? A floor plus operand that wraps
    expect_equal(word, word_type_(0));

    word_type_ bits = make_pattern<word_type_>(0xF0F0);
    atomic_ref_<word_type_> bits_reference(bits);
    bits_reference.clear_bits(word_type_(0xF0), std::memory_order_release);
    bits_reference.set_bits(word_type_(0x0F), std::memory_order_relaxed);
    expect_equal(bits, make_pattern<word_type_>(0xF00F));
    bits_reference.flip_bits(make_pattern<word_type_>(0xFF00), std::memory_order_release);
    expect_equal(bits, make_pattern<word_type_>(0x0F0F));
    expect_equal(bits_reference.fetch_test_and_set_bit(4u, std::memory_order_acq_rel), false);
    expect_equal(bits_reference.fetch_test_and_set_bit(4u, std::memory_order_acquire),
                 true); // ? The second claim loses
    expect_equal(bits_reference.fetch_test_and_reset_bit(0u, std::memory_order_release), true);
    expect_equal(bits_reference.fetch_test_and_reset_bit(0u, std::memory_order_relaxed), false);
    expect_equal(bits, make_pattern<word_type_>(0x0F1E));
    unsigned const top_bit = sizeof(word_type_) * 8 - 1;
    expect_equal(bits_reference.fetch_test_and_set_bit(top_bit, std::memory_order_acq_rel), false);
    expect_equal(bits_reference.fetch_test_and_reset_bit(top_bit, std::memory_order_acq_rel), true);
    expect_equal(bits, make_pattern<word_type_>(0x0F1E)); // ? The top bit round-trips, leaving the rest alone

    using signed_word_t = std::make_signed_t<word_type_>;
    signed_word_t signed_word = -5;
    atomic_ref_<signed_word_t> signed_reference(signed_word);
    expect_equal(signed_reference.fetch_max(signed_word_t(-9), std::memory_order_acquire), signed_word_t(-5));
    expect_equal(signed_reference.fetch_sub_if_at_least(signed_word_t(3), signed_word_t(-8), std::memory_order_acq_rel),
                 signed_word_t(-5));
    expect_equal(signed_reference.fetch_sub_if_at_least(signed_word_t(1), signed_word_t(-8), std::memory_order_acq_rel),
                 signed_word_t(-8)); // ? -8 - 1 < -8: refuses
    expect_equal(signed_reference.fetch_max(signed_word_t(2), std::memory_order_acq_rel),
                 signed_word_t(-8)); // ? Wins only under a signed compare
    expect_equal(signed_reference.fetch_min(signed_word_t(-3), std::memory_order_relaxed), signed_word_t(2));
    expect_equal(signed_word, signed_word_t(-3));

    // The relaxed no-return forms, the ones a far-atomic reference sends past the core, each a
    // distinct operation code there: extrema must compare signed, and the add and bit forms land.
    signed_reference.store_max(signed_word_t(1), std::memory_order_relaxed);
    expect_equal(signed_word, signed_word_t(1)); // ? Wins only under a signed compare
    signed_reference.store_min(signed_word_t(-7), std::memory_order_relaxed);
    expect_equal(signed_word, signed_word_t(-7)); // ? Wins only under a signed compare
    signed_reference.store_add(signed_word_t(20), std::memory_order_relaxed);
    signed_reference.clear_bits(signed_word_t(0x1), std::memory_order_relaxed);
    signed_reference.flip_bits(signed_word_t(0x3), std::memory_order_relaxed);
    expect_equal(signed_word, signed_word_t(15)); // ? 13 loses its low bit, then 0b1100 ^ 0b0011
    signed_reference.store(signed_word_t(-3), std::memory_order_relaxed);

    // The sum must stay at least the floor: a counter at -3 refuses +5 against a floor of 5, as the
    // capacity release needs, where a test of the old value alone would admit it.
    expect_equal(signed_reference.fetch_add_if_at_least(signed_word_t(5), signed_word_t(5), std::memory_order_release),
                 signed_word_t(-3));
    signed_reference.store(signed_word_t(0), std::memory_order_relaxed);
    expect_equal(signed_reference.fetch_add_if_at_least(signed_word_t(5), signed_word_t(5), std::memory_order_release),
                 signed_word_t(0)); // 0 + 5 >= 5: adds
    expect_equal(signed_word, signed_word_t(5));
    constexpr signed_word_t lowest_k = (std::numeric_limits<signed_word_t>::min)();
    constexpr signed_word_t highest_k = (std::numeric_limits<signed_word_t>::max)();
    signed_reference.store(lowest_k, std::memory_order_relaxed);
    expect_equal(signed_reference.fetch_add_if_at_least(signed_word_t(3), signed_word_t(lowest_k + 1),
                                                        std::memory_order_acq_rel),
                 lowest_k); // ? `floor - operand` below the minimum: every word qualifies
    expect_equal(signed_reference.fetch_add_if_at_least(signed_word_t(-1), highest_k, std::memory_order_acq_rel),
                 signed_word_t(lowest_k + 3)); // ? `floor - operand` past the maximum: no word qualifies
    expect_equal(signed_word, signed_word_t(lowest_k + 3));
    word = word_type_(2);
    expect_equal(reference.fetch_add_if_at_least(word_type_(3), word_type_(6), std::memory_order_release),
                 word_type_(2)); // 2 + 3 < 6: refuses
    expect_equal(reference.fetch_add_if_at_least(word_type_(4), word_type_(3), std::memory_order_release),
                 word_type_(2)); // ? The floor below the operand admits every word
    expect_equal(word, word_type_(6));
}

/** A scalar wrapping a @c float bit for bit behind a @c raw_t, as NumKong's do. */
struct wrapped_f32_t {
    using raw_t = float;
    raw_t raw_ = 0;
    constexpr wrapped_f32_t() noexcept = default;
    constexpr explicit wrapped_f32_t(double value) noexcept : raw_(static_cast<raw_t>(value)) {}
    friend constexpr bool operator==(wrapped_f32_t, wrapped_f32_t) noexcept = default;
    friend constexpr bool operator<(wrapped_f32_t left, wrapped_f32_t right) noexcept { return left.raw_ < right.raw_; }
    friend constexpr bool operator>(wrapped_f32_t left, wrapped_f32_t right) noexcept { return left.raw_ > right.raw_; }
};

/** A scalar wrapping a signed 32-bit integer, as NumKong's @c i32_t does. */
struct wrapped_i32_t {
    std::int32_t raw_ = 0;
    constexpr wrapped_i32_t() noexcept = default;
    constexpr wrapped_i32_t(std::int32_t value) noexcept : raw_(value) {}
    friend constexpr auto operator<=>(wrapped_i32_t, wrapped_i32_t) noexcept = default;
    friend constexpr wrapped_i32_t operator+(wrapped_i32_t left, wrapped_i32_t right) noexcept {
        return left.raw_ + right.raw_;
    }
    friend constexpr wrapped_i32_t operator-(wrapped_i32_t left, wrapped_i32_t right) noexcept {
        return left.raw_ - right.raw_;
    }
};

/** A byte-wide minifloat with no @c std::numeric_limits, which the references refuse. */
struct wrapped_e4m3_t {
    std::uint8_t raw_ = 0;
};

template <>
struct std::numeric_limits<wrapped_f32_t> : std::numeric_limits<float> {};
template <>
struct std::numeric_limits<wrapped_i32_t> : std::numeric_limits<std::int32_t> {};

/*  The references admit a scalar by its limits alone, never by its spelling. */
static_assert(fu::atomic_integer<wrapped_i32_t> && !fu::atomic_float<wrapped_i32_t>);
static_assert(fu::atomic_float<wrapped_f32_t> && !fu::atomic_integer<wrapped_f32_t>);
static_assert(!fu::atomic_scalar<wrapped_e4m3_t> && !fu::atomic_integer<bool>);

/** An integer wrapper takes the instructions of the word it wraps, the signed ones included. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref_wrapped() noexcept {
    check_atomic_ref_words<atomic_ref_, wrapped_i32_t>();
    wrapped_i32_t word = -5;
    atomic_ref_<wrapped_i32_t> reference(word);
    expect_equal(reference.fetch_max(wrapped_i32_t(2), std::memory_order_acq_rel),
                 wrapped_i32_t(-5)); // ? Wins only under a signed compare
    reference.store_min(wrapped_i32_t(-7), std::memory_order_relaxed);
    reference.store_max(wrapped_i32_t(1), std::memory_order_release);
    expect_equal(reference.fetch_add_if_at_most(wrapped_i32_t(3), wrapped_i32_t(4), std::memory_order_acq_rel),
                 wrapped_i32_t(1)); // 1 + 3 <= 4: adds
    expect_equal(reference.fetch_add_if_at_least(wrapped_i32_t(-6), wrapped_i32_t(-1), std::memory_order_acq_rel),
                 wrapped_i32_t(4)); // 4 - 6 < -1: refuses
    expect_equal(reference.fetch_sub_if_at_least(wrapped_i32_t(5), wrapped_i32_t(-1), std::memory_order_acq_rel),
                 wrapped_i32_t(4)); // 4 - 5 >= -1: subtracts
    expect_equal(word, wrapped_i32_t(-1));
}

/** The extrema over floats, across the sign where the integer instructions change roles. */
template <template <typename> class atomic_ref_, typename float_type_>
static void check_atomic_ref_floats() noexcept {
    float_type_ word = float_type_(-2.5);
    atomic_ref_<float_type_> reference(word);
    expect_equal(reference.fetch_max(float_type_(-4), std::memory_order_acq_rel), float_type_(-2.5)); // ? Loses
    expect_equal(reference.fetch_max(float_type_(-1), std::memory_order_acq_rel), float_type_(-2.5));
    expect_equal(reference.fetch_max(float_type_(0.5), std::memory_order_relaxed), float_type_(-1));
    expect_equal(reference.fetch_min(float_type_(1), std::memory_order_relaxed), float_type_(0.5)); // ? Loses
    expect_equal(reference.fetch_min(float_type_(-0.25), std::memory_order_acquire), float_type_(0.5));
    expect_equal(reference.fetch_min(float_type_(-3), std::memory_order_acquire), float_type_(-0.25));
    reference.store_max(float_type_(2), std::memory_order_release);
    reference.store_max(float_type_(-7), std::memory_order_relaxed);
    expect_equal(word, float_type_(2));
    reference.store_min(float_type_(-6), std::memory_order_release);
    reference.store_min(float_type_(-5), std::memory_order_relaxed);
    expect_equal(word, float_type_(-6));
}

/** The free functions post through any reference: the standard's with the result discarded, a
 *  ForkUnion one through its own verb, and the bounded forms refuse exactly at their bounds. */
static void test_atomic_verbs_post_through_any_reference() noexcept {
    std::uint32_t word = 5;
    fu::atomic_add(std::atomic_ref<std::uint32_t>(word), 4u, std::memory_order_release);
    fu::atomic_sub(fu::standard_atomic_ref<std::uint32_t>(word), 2u, std::memory_order_relaxed);
    fu::atomic_set_bits(std::atomic_ref<std::uint32_t>(word), 8u, std::memory_order_relaxed);
    fu::atomic_clear_bits(fu::standard_atomic_ref<std::uint32_t>(word), 1u, std::memory_order_relaxed);
    expect_equal(word, 14u); // ? 5 + 4 - 2 = 7, then 7 | 8 = 15, then 15 & ~1 = 14
    fu::atomic_flip_bits(std::atomic_ref<std::uint32_t>(word), 6u, std::memory_order_relaxed);
    fu::atomic_flip_bits(fu::standard_atomic_ref<std::uint32_t>(word), 6u, std::memory_order_relaxed);
    expect_equal(word, 14u); // ? 14 ^ 6 ^ 6, once through each spelling
    expect_equal(
        fu::atomic_fetch_add_if_at_most(std::atomic_ref<std::uint32_t>(word), 2u, 16u, std::memory_order_acq_rel), 14u);
    expect_equal(fu::atomic_fetch_add_if_at_most(fu::standard_atomic_ref<std::uint32_t>(word), 1u, 16u,
                                                 std::memory_order_acq_rel),
                 16u);
    expect_equal(
        fu::atomic_fetch_sub_if_at_least(std::atomic_ref<std::uint32_t>(word), 16u, 0u, std::memory_order_acquire),
        16u);
    expect_equal(fu::atomic_fetch_sub_if_at_least(fu::standard_atomic_ref<std::uint32_t>(word), 1u, 0u,
                                                  std::memory_order_acquire),
                 0u);
    expect_equal(
        fu::atomic_fetch_add_if_at_least(std::atomic_ref<std::uint32_t>(word), 3u, 4u, std::memory_order_release),
        0u); // 0 + 3 < 4: refuses
    expect_equal(fu::atomic_fetch_add_if_at_least(fu::standard_atomic_ref<std::uint32_t>(word), 4u, 4u,
                                                  std::memory_order_release),
                 0u);
    fu::atomic_max(std::atomic_ref<std::uint32_t>(word), 9u, std::memory_order_relaxed);
    fu::atomic_min(fu::standard_atomic_ref<std::uint32_t>(word), 7u, std::memory_order_relaxed);
    expect_equal(word, 7u); // ? 4 rose to 9 through the loop, then fell to 7 through the verb
    double score = 0.5;
    fu::atomic_max(std::atomic_ref<double>(score), -1.0, std::memory_order_relaxed);
    fu::atomic_max(fu::standard_atomic_ref<double>(score), 1.5, std::memory_order_relaxed);
    fu::atomic_min(std::atomic_ref<double>(score), 0.75, std::memory_order_relaxed);
    expect_equal(score, 0.75);
    std::uint32_t flags = 0;
    expect_equal(
        fu::atomic_fetch_test_and_set_bit(std::atomic_ref<std::uint32_t>(flags), 5u, std::memory_order_acq_rel), false);
    expect_equal(
        fu::atomic_fetch_test_and_set_bit(fu::standard_atomic_ref<std::uint32_t>(flags), 5u, std::memory_order_acq_rel),
        true); // ? The verb and the loop agree on the claim
    expect_equal(
        fu::atomic_fetch_test_and_reset_bit(std::atomic_ref<std::uint32_t>(flags), 5u, std::memory_order_release),
        true);
    expect_equal(flags, 0u);
}

/** The shapes the indexes lean on, under contention - a dispenser by the no-return add, a bounded
 *  claim by the conditional add, a high-water mark by @c fetch_max, a lock by @c exchange - each
 *  with an exact expected total. */
template <template <typename> class atomic_ref_>
static void check_atomic_ref_under_contention() noexcept {
    constexpr std::size_t threads_k = 8, rounds_k = 20'000, claim_limit_k = threads_k * rounds_k / 3;
    alignas(fu::default_alignment_k) std::uint32_t dispensed = 0;
    alignas(fu::default_alignment_k) std::uint32_t claimed = 0;
    alignas(fu::default_alignment_k) std::uint64_t high_water = 0;
    alignas(fu::default_alignment_k) bool lock = false;
    alignas(fu::default_alignment_k) std::uint64_t guarded = 0;

    fu::flat_pool_t pool;
    if (!expect_true(fu::succeeded(pool.spawn(threads_k)))) return;
    pool.for_threads([&](std::size_t const thread) noexcept {
        for (std::size_t round = 0; round != rounds_k; ++round) {
            atomic_ref_<std::uint32_t>(dispensed).store_add(1u, std::memory_order_relaxed);
            (void)atomic_ref_<std::uint32_t>(claimed).fetch_add_if_at_most(1u, claim_limit_k,
                                                                           std::memory_order_acq_rel);
            (void)atomic_ref_<std::uint64_t>(high_water)
                .fetch_max(thread * rounds_k + round, std::memory_order_relaxed);
            while (atomic_ref_<bool>(lock).exchange(true, std::memory_order_acquire)) {}
            ++guarded;
            atomic_ref_<bool>(lock).store(false, std::memory_order_release);
        }
    });

    expect_equal(dispensed, threads_k * rounds_k);
    expect_equal(claimed, claim_limit_k); // ? The bounded claim stops exactly at the limit
    expect_equal(high_water, threads_k * rounds_k - 1);
    expect_equal(guarded, threads_k * rounds_k); // ? The lock serialized every increment
}

/** Runs one reference through the contract at every width it spells, where the machine admits it:
 *  the bits it declares match what a CPU class needs before dispatch may pick it, and a missing one
 *  skips the case. */
template <template <typename> class atomic_ref_, typename... word_types_>
static case_result_t test_atomic_ref_keeps_its_contract() noexcept {
    fu::capabilities_t const needed = atomic_ref_<std::uint32_t>::capabilities_k;
    if (needed & ~fu::runtime_capabilities())
        return case_result_t::skipped("needs atomic_ref_<std::uint32_t>::capabilities_k");
    (check_atomic_ref_words<atomic_ref_, word_types_>(), ...);
    (check_atomic_ref_extensions<atomic_ref_, word_types_>(), ...);
    check_atomic_ref_floats<atomic_ref_, float>();
    check_atomic_ref_floats<atomic_ref_, double>();
    check_atomic_ref_floats<atomic_ref_, wrapped_f32_t>();
    check_atomic_ref_wrapped<atomic_ref_>();
    check_atomic_ref_under_contention<atomic_ref_>();
    return {};
}

#else

/** The references need the library's @c std::atomic_ref and @c std::bit_cast; say so rather than
 *  vanish. */
static case_result_t test_atomic_ref_keeps_its_contract() noexcept {
    return case_result_t::skipped("built without __cpp_lib_atomic_ref");
}

#endif // __cpp_lib_atomic_ref && __cpp_lib_bit_cast

int main(int argc, char **argv) {
    start_test_run(argc, argv);

    run_test_case("test_indexed_split_tiles_the_range<>", test_indexed_split_tiles_the_range<>);
    run_test_case("test_coprime_permutation_visits_each_value_once<>",
                  test_coprime_permutation_visits_each_value_once<>);
#if defined(__cpp_lib_atomic_ref) && defined(__cpp_lib_bit_cast)
    run_test_case("test_atomic_ref_keeps_its_contract<fu::standard_atomic_ref, std::uint8_t, std::uint16_t, " //
                  "std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::standard_atomic_ref, std::uint8_t, std::uint16_t,
                                                     std::uint32_t, std::uint64_t>);
    run_test_case("test_atomic_verbs_post_through_any_reference", test_atomic_verbs_post_through_any_reference);
#if FORKUNION_TARGET_X86_CMPCCXADD
    run_test_case("test_atomic_ref_keeps_its_contract<fu::x86_cmpccxadd_atomic_ref, std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::x86_cmpccxadd_atomic_ref, std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_X86_RAOINT
    run_test_case("test_atomic_ref_keeps_its_contract<fu::x86_raoint_atomic_ref, std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::x86_raoint_atomic_ref, std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_ARM64_LSE
    run_test_case("test_atomic_ref_keeps_its_contract<fu::arm64_lse_atomic_ref, std::uint8_t, std::uint16_t, " //
                  "std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::arm64_lse_atomic_ref, std::uint8_t, std::uint16_t,
                                                     std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_ARM64_RCPC
    run_test_case("test_atomic_ref_keeps_its_contract<fu::arm64_rcpc_atomic_ref, std::uint8_t, std::uint16_t, " //
                  "std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::arm64_rcpc_atomic_ref, std::uint8_t, std::uint16_t,
                                                     std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_RISCV64_ATOMIC
    run_test_case("test_atomic_ref_keeps_its_contract<fu::riscv64_atomic_ref, std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::riscv64_atomic_ref, std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_RISCV64_ZACAS
    run_test_case("test_atomic_ref_keeps_its_contract<fu::riscv64_zacas_atomic_ref, std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::riscv64_zacas_atomic_ref, std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_RISCV64_ZABHA
    run_test_case("test_atomic_ref_keeps_its_contract<fu::riscv64_zabha_atomic_ref, std::uint8_t, std::uint16_t, " //
                  "std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::riscv64_zabha_atomic_ref, std::uint8_t, std::uint16_t,
                                                     std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_PPC64_AMO
    run_test_case("test_atomic_ref_keeps_its_contract<fu::ppc64_amo_atomic_ref, std::uint8_t, std::uint16_t, " //
                  "std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::ppc64_amo_atomic_ref, std::uint8_t, std::uint16_t,
                                                     std::uint32_t, std::uint64_t>);
#endif
#if FORKUNION_TARGET_WASM_WAIT
    run_test_case("test_atomic_ref_keeps_its_contract<fu::wasm_atomic_ref, std::uint32_t, std::uint64_t>",
                  test_atomic_ref_keeps_its_contract<fu::wasm_atomic_ref, std::uint32_t, std::uint64_t>);
#endif
#else
    run_test_case("test_atomic_ref_keeps_its_contract", test_atomic_ref_keeps_its_contract);
#endif
    run_test_case("test_topology_counts_and_core_partition", test_topology_counts_and_core_partition);
    run_test_case("test_replicated_array_gives_every_domain_a_copy", test_replicated_array_gives_every_domain_a_copy);
    run_test_case("test_sharded_array_round_trips_its_segments", test_sharded_array_round_trips_its_segments);
    run_test_case("test_spawn_rejects_zero_threads", test_spawn_rejects_zero_threads);
    run_test_case("test_spawn_succeeds_at_full_width<>", test_spawn_succeeds_at_full_width<>);
    run_test_case("test_caller_exclusivity_is_queried_not_cached<>", test_caller_exclusivity_is_queried_not_cached<>);
    run_test_case("test_for_threads_reaches_every_worker<>", test_for_threads_reaches_every_worker<>);
    run_test_case("test_unsafe_for_threads_reaches_every_worker<>", test_unsafe_for_threads_reaches_every_worker<>);
    run_test_case("test_generation_polling_on_an_exclusive_pool<>", test_generation_polling_on_an_exclusive_pool<>);
    run_test_case("test_broadcast_join_dispatches_at_construction<>", test_broadcast_join_dispatches_at_construction<>);
    run_test_case("test_broadcast_join_destructor_joins<>", test_broadcast_join_destructor_joins<>);
    run_test_case("test_generation_inclusive_join_runs_the_caller_slice<>",
                  test_generation_inclusive_join_runs_the_caller_slice<>);
    run_test_case("test_generation_on_a_lone_caller", test_generation_on_a_lone_caller);
    run_test_case("test_generation_on_a_lone_worker", test_generation_on_a_lone_worker);
    run_test_case("test_forks_from_thread_zero<>", test_forks_from_thread_zero<>);
    run_test_case("test_generation_survives_back_to_back_dispatches<>",
                  test_generation_survives_back_to_back_dispatches<>);
    run_test_case("test_mixed_exclusivity_overlaps_through_tokens<>", test_mixed_exclusivity_overlaps_through_tokens<>);
    run_test_case("test_mixed_exclusivity_overlaps_through_guards<>", test_mixed_exclusivity_overlaps_through_guards<>);
    run_test_case("test_for_n_across_uncomfortable_sizes<>", test_for_n_across_uncomfortable_sizes<>);
    run_test_case("test_for_n_visits_each_task_once_and_again<>", test_for_n_visits_each_task_once_and_again<>);
    run_test_case("test_for_slices_partitions_the_range<>", test_for_slices_partitions_the_range<>);
    run_test_case("test_for_n_dynamic_covers_uneven_work<>", test_for_n_dynamic_covers_uneven_work<>);
    run_test_case("test_for_n_dynamic_steals_from_a_stalled_thread<>",
                  test_for_n_dynamic_steals_from_a_stalled_thread<>);
    run_test_case("test_for_n_dynamic_refusals_leave_the_cursors<>", test_for_n_dynamic_refusals_leave_the_cursors<>);
    run_test_case("test_for_n_dynamic_oversubscribed<>", test_for_n_dynamic_oversubscribed<>);
    run_test_case("test_generation_stale_token_stays_complete<>", test_generation_stale_token_stays_complete<>);
    run_test_case("test_sleep_then_wake_loses_nothing<>", test_sleep_then_wake_loses_nothing<>);
    run_test_case("test_two_callers_drive_two_pools_apart", test_two_callers_drive_two_pools_apart);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<false>",
                  test_static_and_dynamic_dispatch_share_a_pool<false>);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<true>",
                  test_static_and_dynamic_dispatch_share_a_pool<true>);
    run_test_case("test_spawn_terminate_churn_loses_nothing<>", test_spawn_terminate_churn_loses_nothing<>);
#if FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN
    run_test_case("test_spawn_succeeds_at_full_width<make_colocated_pool_t>",
                  test_spawn_succeeds_at_full_width<make_colocated_pool_t>);
    run_test_case("test_caller_exclusivity_is_queried_not_cached<make_colocated_pool_t>",
                  test_caller_exclusivity_is_queried_not_cached<make_colocated_pool_t>);
    run_test_case("test_for_threads_reaches_every_worker<make_colocated_pool_t>",
                  test_for_threads_reaches_every_worker<make_colocated_pool_t>);
    run_test_case("test_unsafe_for_threads_reaches_every_worker<make_colocated_pool_t>",
                  test_unsafe_for_threads_reaches_every_worker<make_colocated_pool_t>);
    run_test_case("test_generation_polling_on_an_exclusive_pool<make_colocated_pool_t>",
                  test_generation_polling_on_an_exclusive_pool<make_colocated_pool_t>);
    run_test_case("test_broadcast_join_dispatches_at_construction<make_colocated_pool_t>",
                  test_broadcast_join_dispatches_at_construction<make_colocated_pool_t>);
    run_test_case("test_broadcast_join_destructor_joins<make_colocated_pool_t>",
                  test_broadcast_join_destructor_joins<make_colocated_pool_t>);
    run_test_case("test_generation_inclusive_join_runs_the_caller_slice<make_colocated_pool_t>",
                  test_generation_inclusive_join_runs_the_caller_slice<make_colocated_pool_t>);
    run_test_case("test_forks_from_thread_zero<make_colocated_pool_t>",
                  test_forks_from_thread_zero<make_colocated_pool_t>);
    run_test_case("test_generation_survives_back_to_back_dispatches<make_colocated_pool_t>",
                  test_generation_survives_back_to_back_dispatches<make_colocated_pool_t>);
    run_test_case("test_mixed_exclusivity_overlaps_through_tokens<make_colocated_pool_t>",
                  test_mixed_exclusivity_overlaps_through_tokens<make_colocated_pool_t>);
    run_test_case("test_mixed_exclusivity_overlaps_through_guards<make_colocated_pool_t>",
                  test_mixed_exclusivity_overlaps_through_guards<make_colocated_pool_t>);
    run_test_case("test_for_n_across_uncomfortable_sizes<make_colocated_pool_t>",
                  test_for_n_across_uncomfortable_sizes<make_colocated_pool_t>);
    run_test_case("test_for_n_visits_each_task_once_and_again<make_colocated_pool_t>",
                  test_for_n_visits_each_task_once_and_again<make_colocated_pool_t>);
    run_test_case("test_for_slices_partitions_the_range<make_colocated_pool_t>",
                  test_for_slices_partitions_the_range<make_colocated_pool_t>);
    run_test_case("test_for_n_dynamic_covers_uneven_work<make_colocated_pool_t>",
                  test_for_n_dynamic_covers_uneven_work<make_colocated_pool_t>);
    run_test_case("test_for_n_dynamic_steals_from_a_stalled_thread<make_colocated_pool_t>",
                  test_for_n_dynamic_steals_from_a_stalled_thread<make_colocated_pool_t>);
    run_test_case("test_for_n_dynamic_refusals_leave_the_cursors<make_colocated_pool_t>",
                  test_for_n_dynamic_refusals_leave_the_cursors<make_colocated_pool_t>);
    run_test_case("test_for_n_dynamic_oversubscribed<make_colocated_pool_t>",
                  test_for_n_dynamic_oversubscribed<make_colocated_pool_t>);
    run_test_case("test_generation_stale_token_stays_complete<make_colocated_pool_t>",
                  test_generation_stale_token_stays_complete<make_colocated_pool_t>);
    run_test_case("test_sleep_then_wake_loses_nothing<make_colocated_pool_t>",
                  test_sleep_then_wake_loses_nothing<make_colocated_pool_t>);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<false, make_colocated_pool_t>",
                  test_static_and_dynamic_dispatch_share_a_pool<false, make_colocated_pool_t>);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<true, make_colocated_pool_t>",
                  test_static_and_dynamic_dispatch_share_a_pool<true, make_colocated_pool_t>);
    run_test_case("test_spawn_terminate_churn_loses_nothing<make_colocated_pool_t>",
                  test_spawn_terminate_churn_loses_nothing<make_colocated_pool_t>);
    run_test_case("test_spawn_succeeds_at_full_width<make_distributed_pool_t>",
                  test_spawn_succeeds_at_full_width<make_distributed_pool_t>);
    run_test_case("test_caller_exclusivity_is_queried_not_cached<make_distributed_pool_t>",
                  test_caller_exclusivity_is_queried_not_cached<make_distributed_pool_t>);
    run_test_case("test_for_threads_reaches_every_worker<make_distributed_pool_t>",
                  test_for_threads_reaches_every_worker<make_distributed_pool_t>);
    run_test_case("test_unsafe_for_threads_reaches_every_worker<make_distributed_pool_t>",
                  test_unsafe_for_threads_reaches_every_worker<make_distributed_pool_t>);
    run_test_case("test_generation_polling_on_an_exclusive_pool<make_distributed_pool_t>",
                  test_generation_polling_on_an_exclusive_pool<make_distributed_pool_t>);
    run_test_case("test_broadcast_join_dispatches_at_construction<make_distributed_pool_t>",
                  test_broadcast_join_dispatches_at_construction<make_distributed_pool_t>);
    run_test_case("test_broadcast_join_destructor_joins<make_distributed_pool_t>",
                  test_broadcast_join_destructor_joins<make_distributed_pool_t>);
    run_test_case("test_generation_inclusive_join_runs_the_caller_slice<make_distributed_pool_t>",
                  test_generation_inclusive_join_runs_the_caller_slice<make_distributed_pool_t>);
    run_test_case("test_forks_from_thread_zero<make_distributed_pool_t>",
                  test_forks_from_thread_zero<make_distributed_pool_t>);
    run_test_case("test_generation_survives_back_to_back_dispatches<make_distributed_pool_t>",
                  test_generation_survives_back_to_back_dispatches<make_distributed_pool_t>);
    run_test_case("test_mixed_exclusivity_overlaps_through_tokens<make_distributed_pool_t>",
                  test_mixed_exclusivity_overlaps_through_tokens<make_distributed_pool_t>);
    run_test_case("test_mixed_exclusivity_overlaps_through_guards<make_distributed_pool_t>",
                  test_mixed_exclusivity_overlaps_through_guards<make_distributed_pool_t>);
    run_test_case("test_for_n_across_uncomfortable_sizes<make_distributed_pool_t>",
                  test_for_n_across_uncomfortable_sizes<make_distributed_pool_t>);
    run_test_case("test_for_n_visits_each_task_once_and_again<make_distributed_pool_t>",
                  test_for_n_visits_each_task_once_and_again<make_distributed_pool_t>);
    run_test_case("test_for_slices_partitions_the_range<make_distributed_pool_t>",
                  test_for_slices_partitions_the_range<make_distributed_pool_t>);
    run_test_case("test_for_n_dynamic_covers_uneven_work<make_distributed_pool_t>",
                  test_for_n_dynamic_covers_uneven_work<make_distributed_pool_t>);
    run_test_case("test_for_n_dynamic_steals_from_a_stalled_thread<make_distributed_pool_t>",
                  test_for_n_dynamic_steals_from_a_stalled_thread<make_distributed_pool_t>);
    run_test_case("test_for_n_dynamic_refusals_leave_the_cursors<make_distributed_pool_t>",
                  test_for_n_dynamic_refusals_leave_the_cursors<make_distributed_pool_t>);
    run_test_case("test_for_n_dynamic_covers_every_regime_across_domains<make_distributed_pool_t>",
                  test_for_n_dynamic_covers_every_regime_across_domains<make_distributed_pool_t>);
    run_test_case("test_for_n_dynamic_oversubscribed<make_distributed_pool_t>",
                  test_for_n_dynamic_oversubscribed<make_distributed_pool_t>);
    run_test_case("test_generation_stale_token_stays_complete<make_distributed_pool_t>",
                  test_generation_stale_token_stays_complete<make_distributed_pool_t>);
    run_test_case("test_sleep_then_wake_loses_nothing<make_distributed_pool_t>",
                  test_sleep_then_wake_loses_nothing<make_distributed_pool_t>);
    run_test_case("test_awkward_spawn_shapes_dispatch_once", test_awkward_spawn_shapes_dispatch_once);
    run_test_case("test_fabric_ranks_bandwidth_before_latency", test_fabric_ranks_bandwidth_before_latency);
    run_test_case("test_fabric_harvest_fills_edges", test_fabric_harvest_fills_edges);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<false, make_distributed_pool_t>",
                  test_static_and_dynamic_dispatch_share_a_pool<false, make_distributed_pool_t>);
    run_test_case("test_static_and_dynamic_dispatch_share_a_pool<true, make_distributed_pool_t>",
                  test_static_and_dynamic_dispatch_share_a_pool<true, make_distributed_pool_t>);
    run_test_case("test_spawn_terminate_churn_loses_nothing<make_distributed_pool_t>",
                  test_spawn_terminate_churn_loses_nothing<make_distributed_pool_t>);
#if FORKUNION_OS_LINUX_ && FORKUNION_WITH_PLACE_THREADS_BY_AFFINITY
    run_test_case("test_caller_affinity_is_preserved", test_caller_affinity_is_preserved);
#endif
#endif // FORKUNION_WITH_COLOCATE_POOLS_ON_DOMAIN

    // On 32-bit architectures, limit thread counts to avoid resource exhaustion. Each thread needs
    // ~8MB stack, and 255 threads would consume 2GB+ address space
    std::size_t const max_cores = fu::allowed_cores_count();
    constexpr std::size_t max_stress_threads = sizeof(void *) == 4 ? 23 : 255;
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(3, 3, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 3, 3, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(3, 2, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 3, 2, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(3, 4, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 3, 4, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(3, 5, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 3, 5, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(7, max_stress_threads, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 7, max_stress_threads, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(max_stress_threads, 7, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, max_stress_threads, 7, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(max_stress_threads - 2, max_stress_threads - 1, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, max_stress_threads - 2, max_stress_threads - 1, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(max_stress_threads - 2, max_stress_threads, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, max_stress_threads - 2, max_stress_threads, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(max_stress_threads, max_stress_threads, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, max_stress_threads, max_stress_threads, 1);
    // ? 640 epochs wrap the `uint8` clock twice, and 132K epochs the `uint16` one
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu8_t>(3, 5, 160)",
                  test_mixed_dispatch_survives_epoch_wraps<fu8_t>, 3, 5, 160);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu16_t>(3, 5, 33000)",
                  test_mixed_dispatch_survives_epoch_wraps<fu16_t>, 3, 5, 33000);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu16_t>(max_cores, UINT16_MAX, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu16_t>, max_cores, UINT16_MAX, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu16_t>(max_stress_threads, UINT16_MAX, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu16_t>, max_stress_threads, UINT16_MAX, 1);
    run_test_case("test_mixed_dispatch_survives_epoch_wraps<fu32_t>(max_cores, 100000, 1)",
                  test_mixed_dispatch_survives_epoch_wraps<fu32_t>, max_cores, 100000, 1);
    return finish_test_run();
}
