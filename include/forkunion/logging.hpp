/**
 *  @brief Human-readable dumps of the harvested topology and capabilities.
 *  @author Ash Vardanian
 *  @file include/forkunion/logging.hpp
 *  @date July 10, 2026
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "topology.hpp"

namespace ashvardanian {
namespace forkunion {

/**
 *  @brief Detects if the output stream supports ANSI color codes.
 *  @note Every accessor below returns its escape sequence when colors are on, and an empty string when they
 *      are off, so call sites interpolate them unconditionally.
 */
struct logging_colors_t {
    /** Whether the accessors emit escape sequences or empty strings. */
    bool use_colors_ = false;

    /**
     *  @brief Forces coloring on or off, bypassing the terminal probe.
     *  @param[in] use_colors Whether the accessors should emit ANSI escape sequences.
     */
    explicit logging_colors_t(bool use_colors) noexcept : use_colors_(use_colors) {}

    /**
     *  @brief Probes `stdout` and enables coloring only for a terminal advertising it.
     *  @note On POSIX the `TERM` variable must name a color-capable terminal; Windows consoles are assumed capable.
     */
    explicit logging_colors_t() noexcept {
#if FU_ON_WINDOWS
        if (!::_isatty(_fileno(stdout))) return;
#endif
#if FU_ON_POSIX
        if (!::isatty(STDOUT_FILENO)) return;
#endif
#if FU_ON_WINDOWS
        // On Windows, assume color support is available
        use_colors_ = true;
#else
        char const *term = std::getenv("TERM");
        if (!term) return;
        use_colors_ = std::strstr(term, "color") != nullptr || std::strstr(term, "xterm") != nullptr ||
                      std::strstr(term, "screen") != nullptr || std::strcmp(term, "linux") == 0;
#endif
    }

    /* ANSI style codes */
    char const *reset() const noexcept { return use_colors_ ? "\033[0m" : ""; }
    char const *bold() const noexcept { return use_colors_ ? "\033[1m" : ""; }
    char const *dim() const noexcept { return use_colors_ ? "\033[2m" : ""; }

    /* ANSI color codes */
    char const *red() const noexcept { return use_colors_ ? "\033[31m" : ""; }
    char const *green() const noexcept { return use_colors_ ? "\033[32m" : ""; }
    char const *yellow() const noexcept { return use_colors_ ? "\033[33m" : ""; }
    char const *blue() const noexcept { return use_colors_ ? "\033[34m" : ""; }
    char const *magenta() const noexcept { return use_colors_ ? "\033[35m" : ""; }
    char const *cyan() const noexcept { return use_colors_ ? "\033[36m" : ""; }
    char const *white() const noexcept { return use_colors_ ? "\033[37m" : ""; }
    char const *gray() const noexcept { return use_colors_ ? "\033[90m" : ""; }

    /* Compound styles */
    char const *bold_red() const noexcept { return use_colors_ ? "\033[1;31m" : ""; }
    char const *bold_green() const noexcept { return use_colors_ ? "\033[1;32m" : ""; }
    char const *bold_yellow() const noexcept { return use_colors_ ? "\033[1;33m" : ""; }
    char const *bold_blue() const noexcept { return use_colors_ ? "\033[1;34m" : ""; }
    char const *bold_magenta() const noexcept { return use_colors_ ? "\033[1;35m" : ""; }
    char const *bold_cyan() const noexcept { return use_colors_ ? "\033[1;36m" : ""; }
    char const *bold_white() const noexcept { return use_colors_ ? "\033[1;37m" : ""; }
    char const *bold_gray() const noexcept { return use_colors_ ? "\033[1;90m" : ""; }
};

/** Formats memory volume in @p bytes with appropriate units and precision, like @b "1.5 GiB". */
struct log_memory_volume_t {

    /**
     *  @brief Prints @p bytes into @p buffer with binary units, switching at every 1024-fold boundary.
     *  @param[in] bytes Volume to format, rendered with one decimal place from a KiB upwards.
     *  @param[out] buffer Destination for the NUL-terminated string.
     *  @param[in] buffer_size Capacity of @p buffer in bytes, including the terminator.
     *  @param[in] colors Palette tinting the number and its unit.
     *  @note Output is truncated to fit and always NUL-terminated; escape sequences count against the budget,
     *      so allow 64 bytes for colored output.
     */
    void operator()(std::size_t bytes, char *buffer, std::size_t buffer_size, logging_colors_t colors) const noexcept {

        char const *value_color = colors.bold_white();
        char const *unit_color = colors.dim();
        char const *reset_color = colors.reset();

        if (bytes >= (1ull << 40)) {
            double tb = static_cast<double>(bytes) / (1ull << 40);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sTiB%s", value_color, tb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 30)) {
            double gb = static_cast<double>(bytes) / (1ull << 30);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sGiB%s", value_color, gb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 20)) {
            double mb = static_cast<double>(bytes) / (1ull << 20);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sMiB%s", value_color, mb, unit_color, unit_color,
                          reset_color);
        }
        else if (bytes >= (1ull << 10)) {
            double kb = static_cast<double>(bytes) / (1ull << 10);
            std::snprintf(buffer, buffer_size, "%s%.1f%s %sKiB%s", value_color, kb, unit_color, unit_color,
                          reset_color);
        }
        else {
            std::snprintf(buffer, buffer_size, "%s%zu%s %sB%s", value_color, bytes, unit_color, unit_color,
                          reset_color);
        }
    }
};

/** Formats a set of CPU core IDs in a compact and readable way, like @b "0-3,5,7,8,10-12". */
struct log_core_range_t {

    /**
     *  @brief Prints @p count core IDs into @p buffer, collapsing a contiguous run into a range.
     *  @param[in] core_ids Core IDs to format, expected in ascending order; may be empty.
     *  @param[in] count Number of entries in @p core_ids.
     *  @param[out] buffer Destination for the NUL-terminated string.
     *  @param[in] buffer_size Capacity of @p buffer in bytes, including the terminator.
     *  @param[in] colors Palette tinting the numbers.
     *  @note An empty set prints "none", a contiguous one prints "first-last", and beyond 8 cores the middle
     *      is elided with an ellipsis.
     *  @note Output is truncated to fit and always NUL-terminated; allow 256 bytes for colored output.
     */
    void operator()(                                  //
        core_id_t const *core_ids, std::size_t count, //
        char *buffer, std::size_t buffer_size, logging_colors_t colors) const noexcept {

        if (count == 0) {
            std::snprintf(buffer, buffer_size, "%snone%s", colors.dim(), colors.reset());
            return;
        }

        char const *value_color = colors.bold_white();
        char const *reset_color = colors.reset();

        if (count == 1) {
            std::snprintf(buffer, buffer_size, "%s%d%s", value_color, core_ids[0], reset_color);
            return;
        }

        // Check if it's a contiguous range
        bool is_contiguous = true;
        for (std::size_t i = 1; i < count && is_contiguous; ++i)
            if (core_ids[i] != core_ids[i - 1] + 1) is_contiguous = false;

        if (is_contiguous) {
            std::snprintf(                            //
                buffer, buffer_size, "%s%d%s-%s%d%s", //
                value_color, core_ids[0], reset_color, value_color, core_ids[count - 1], reset_color);
        }
        else {
            // Show first few and last few with ellipsis if many cores
            if (count <= 8) {
                int written = std::snprintf(buffer, buffer_size, "%s%d%s", value_color, core_ids[0], reset_color);
                for (std::size_t i = 1; i < count && written < static_cast<int>(buffer_size) - 1; ++i)
                    written += std::snprintf(                                                         //
                        buffer + written, buffer_size - static_cast<std::size_t>(written), ",%s%d%s", //
                        value_color, core_ids[i], reset_color);
            }
            else {
                std::snprintf(                                                        //
                    buffer, buffer_size, "%s%d%s,%s%d%s,%s%d%s…%s%d%s,%s%d%s,%s%d%s", //
                    value_color, core_ids[0], reset_color, value_color, core_ids[1], reset_color, value_color,
                    core_ids[2], reset_color, value_color, core_ids[count - 3], reset_color, value_color,
                    core_ids[count - 2], reset_color, value_color, core_ids[count - 1], reset_color);
            }
        }
    }
};

/** NUMA topology logger with compact tree design and color support. */
struct log_numa_topology_t {

    /**
     *  @brief Logs NUMA topology in compact tree format with colors.
     *  @param[in] topology The harvested topology whose sockets, domains, cores, and page sizes are printed.
     *  @param[in] colors Whether to emit ANSI colour codes, and which.
     *  @param[in] output Destination stream, defaulting to `stdout`.
     *  @note An empty topology prints "No NUMA nodes detected", and only page sizes above 4 KiB are listed.
     *  @note Each row is assembled in a 1024-byte line buffer, so an unusually wide row is truncated.
     */
    template <std::size_t max_page_sizes_, typename allocator_type_>
    void operator()(machine_topology<max_page_sizes_, allocator_type_> const &topology, logging_colors_t colors,
                    std::FILE *output = stdout) const noexcept {

        // Line buffer for assembly
        char line_buffer[1024];
        logging_colors_t colorless {false};

        // Helper lambda to flush line buffer
        auto flush_line = [&]() { std::fprintf(output, "%s", line_buffer); };

        // Main header
        std::snprintf(line_buffer, sizeof(line_buffer), "%sNUMA Layout%s\n", colors.bold_cyan(), colors.reset());
        flush_line();

        if (topology.memory_domains_count() == 0) {
            std::snprintf(line_buffer, sizeof(line_buffer), "%sNo NUMA nodes detected%s\n", colors.dim(),
                          colors.reset());
            flush_line();
            return;
        }

        // Get the last socket ID for comparison
        int last_socket_id =
            topology.memory_domain_at(static_cast<memory_domain_index_t>(topology.memory_domains_count() - 1))
                .socket_id;
        int current_socket_id = -1;

        for (std::size_t i = 0; i < topology.memory_domains_count(); ++i) {
            auto const node = topology.memory_domain_at(static_cast<memory_domain_index_t>(i));

            // Print socket header when we encounter a new socket
            if (node.socket_id != current_socket_id) {
                current_socket_id = node.socket_id;
                bool is_last_socket = current_socket_id == last_socket_id;

                std::snprintf(                                                     //
                    line_buffer, sizeof(line_buffer), "%s%s─ %sSocket%s %s%d%s\n", //
                    colors.dim(), is_last_socket ? "└" : "├",                      //
                    colors.blue(), /* "Socket" */ colors.reset(),                  //
                    colors.bold_blue(), current_socket_id, colors.reset());
                flush_line();
            }

            // Check if this is the last node in current socket
            bool is_last_node_in_socket =
                (i + 1 >= topology.memory_domains_count() ||
                 topology.memory_domain_at(static_cast<memory_domain_index_t>(i + 1)).socket_id != current_socket_id);

            // Format core range and memory
            char cores_str[256], memory_str[64];
            log_core_range_t {}(node.first_core_id, node.logical_cores_count, cores_str, sizeof(cores_str), colorless);
            log_memory_volume_t {}(node.volume_ram, memory_str, sizeof(memory_str), colorless);

            // Tree structure prefixes
            bool is_last_socket = current_socket_id == last_socket_id;
            char const *socket_prefix = is_last_socket ? "   " : "│  ";
            char const *node_connector = is_last_node_in_socket ? "└─ " : "├─ ";

            // Start building node line
            int pos = std::snprintf(                                                      //
                line_buffer, sizeof(line_buffer),                                         //
                "%s%s%s%sNode%s %s%d%s • %sCores:%s %s%s (%zu)%s • %sMemory:%s %s%s%s",   //
                colors.dim(), socket_prefix, node_connector,                              //
                colors.cyan(), /* "Node" */ colors.reset(),                               //
                colors.bold_cyan(), node.memory_domain_id, colors.reset(),                //
                colors.green(), /* "Cores:" */ colors.reset(),                            //
                colors.bold_green(), cores_str, node.logical_cores_count, colors.reset(), //
                colors.yellow(), /* "Memory:" */ colors.reset(),                          //
                colors.bold_yellow(), memory_str, colors.reset());

            // Add huge pages if any exist
            auto const &page_settings = node.page_sizes;
            bool first_page = true;

            for (std::size_t j = 0; j < page_settings.size(); ++j) {
                auto const &ps = page_settings[j];
                if (ps.bytes_per_page <= 4096) continue; // Skip regular pages

                if (first_page) {
                    pos += static_cast<std::size_t>(std::snprintf(                      //
                        line_buffer + pos, sizeof(line_buffer) - pos, " • %sPages:%s ", //
                        colors.magenta(), /* "Pages:" */ colors.reset()));
                    first_page = false;
                }
                else pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, " "));

                char page_size_str[32], page_volume_str[32];
                std::size_t free_bytes = ps.free_pages * ps.bytes_per_page;
                log_memory_volume_t {}(ps.bytes_per_page, page_size_str, sizeof(page_size_str), colorless);
                log_memory_volume_t {}(free_bytes, page_volume_str, sizeof(page_volume_str), colorless);

                pos += static_cast<std::size_t>(std::snprintf(                   //
                    line_buffer + pos, sizeof(line_buffer) - pos, "%s%s (%s)%s", //
                    colors.bold_magenta(), page_size_str, page_volume_str, colors.reset()));
            }

            std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n");
            flush_line();
        }

        // Final newline
        std::snprintf(line_buffer, sizeof(line_buffer), "\n");
        flush_line();
    }
};

/** Logs the CPU and memory capabilities as a two-row tree, one bullet per recognized bit. */
struct log_capabilities_t {

    /** One bullet of a row: the bit that earns it and the label it prints as. */
    struct bullet_t {
        capabilities_t bit;
        char const *label;
    };

    /**
     *  @brief Logs the detected capability bits as a two-row tree, one row for the CPU and one for the RAM.
     *  @param[in] caps Bit-mask to render, where every recognized bit becomes one bullet.
     *  @param[in] colors Whether to emit ANSI colour codes, and which.
     *  @param[in] output Destination stream, defaulting to `stdout`.
     *  @note Only the instruction and memory-placement bits are listed; a row matching none prints "None detected".
     */
    void operator()(capabilities_t caps, logging_colors_t colors, std::FILE *output = stdout) const noexcept {

        // One row of the tree: the branch glyph and title, then every present bit as a bullet in the
        // row's tint, or a dim placeholder when none is.
        auto print_row = [&](char const *branch, char const *title, char const *tint, auto const &bullets) noexcept {
            std::fprintf(output, "%s%s %s%s:%s ", colors.dim(), branch, colors.cyan(), title, colors.reset());
            bool first = true;
            for (bullet_t const &bullet : bullets) {
                if (!(caps & bullet.bit)) continue;
                std::fprintf(output, "%s%s%s%s", first ? "" : " • ", tint, bullet.label, colors.reset());
                first = false;
            }
            if (first) std::fprintf(output, "%sNone detected%s", colors.dim(), colors.reset());
            std::fprintf(output, "\n");
        };

        constexpr bullet_t cpu_bullets[] = {
            {capability_x86_pause_k, "x86 PAUSE"},         {capability_x86_tpause_k, "x86 TPAUSE"},
            {capability_x86_cmpccxadd_k, "x86 CMPCCXADD"}, {capability_x86_raoint_k, "x86 RAO-INT"},
            {capability_arm64_yield_k, "ARM64 YIELD"},     {capability_arm64_wfet_k, "ARM64 WFET"},
            {capability_arm64_lse_k, "ARM64 LSE"},         {capability_arm64_rcpc_k, "ARM64 RCPC"},
            {capability_risc5_pause_k, "RISC-V PAUSE"},    {capability_risc5_wrs_k, "RISC-V WRS"},
            {capability_risc5_zacas_k, "RISC-V ZACAS"},
        };
        constexpr bullet_t ram_bullets[] = {
            {capability_place_memory_on_domain_k, "NUMA"},
            {capability_place_huge_pages_on_domain_k, "Huge Pages"},
            {capability_huge_transparent_pages_k, "Transparent Huge Pages"},
        };

        std::fprintf(output, "%sSystem Capabilities%s\n", colors.bold_cyan(), colors.reset());
        print_row("├─", "CPU", colors.bold_green(), cpu_bullets);
        print_row("└─", "RAM", colors.bold_yellow(), ram_bullets);
        std::fprintf(output, "\n");
    }
};

} // namespace forkunion
} // namespace ashvardanian
