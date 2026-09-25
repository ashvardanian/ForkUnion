/**
 *  @file bench/harness.hpp
 *  @author Ash Vardanian
 *  @date September 24, 2026
 *  @brief The environment reader and the opening lines the C++ benchmarks share.
 */
#pragma once
#include <charconv>     // `std::from_chars`
#include <cstdio>       // `std::fprintf`, `std::printf`
#include <cstdlib>      // `std::getenv`, `std::strtod`, `std::abort`
#include <cstring>      // `std::strcmp`, `std::strlen`
#include <system_error> // `std::errc`
#include <type_traits>  // `std::is_same_v`, `std::is_floating_point_v`

#include <forkunion.hpp> // `FORKUNION_VERSION_*`, `capability_name`

namespace ashvardanian::forkunion::bench {

/** Prints @p label, then each bit of @p capabilities by its @c capability_name, comma-separated. */
inline void log_capabilities(char const *label, capabilities_t const capabilities) noexcept {
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

/** Prints the version, the capabilities this build carries, and the ones this machine offers. */
inline void log_environment() noexcept {
    std::printf("ForkUnion %d.%d.%d\n", FORKUNION_VERSION_MAJOR, FORKUNION_VERSION_MINOR, FORKUNION_VERSION_PATCH);
    log_capabilities("- Compiled for: ", comptime_capabilities());
    log_capabilities("- This machine: ", runtime_capabilities());
}

/** Reads the environment variable @p name as @p value_type_, or returns @p fallback when it is
 *  unset or empty; aborts naming the variable when the text does not parse, so a typo never
 *  becomes a silent default. */
template <typename value_type_>
[[nodiscard]] value_type_ env_variable(char const *name, value_type_ fallback) noexcept {
    char const *const text = std::getenv(name);
    if (!text || !*text) return fallback;
    if constexpr (std::is_same_v<value_type_, char const *>) return text;
    else if constexpr (std::is_same_v<value_type_, bool>) return std::strcmp(text, "0") && std::strcmp(text, "false");
    else {
        value_type_ value {};
        char *stop = nullptr;
        if constexpr (std::is_floating_point_v<value_type_>) value = static_cast<value_type_>(std::strtod(text, &stop));
        else {
            auto const [end, error] = std::from_chars(text, text + std::strlen(text), value);
            stop = error == std::errc {} ? const_cast<char *>(end) : const_cast<char *>(text);
        }
        if (stop != text && *stop == '\0') return value;
        std::fprintf(stderr, "%s=\"%s\" does not parse\n", name, text);
        std::abort();
    }
}

} // namespace ashvardanian::forkunion::bench
