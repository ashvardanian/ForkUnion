/**
 *  @file logging.hpp
 *  @brief Human-readable dumps of the harvested topology and capabilities.
 *  @note Included by `<forkunion.hpp>`; not meant to be included on its own.
 */
#pragma once
#include "topology.hpp"

namespace ashvardanian {
namespace forkunion {

/**
 *  @brief Detects if the output stream supports ANSI color codes.
 */
struct logging_colors_t {
    bool use_colors_ = false;

    explicit logging_colors_t(bool use_colors) noexcept : use_colors_(use_colors) {}

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

/**
 *  @brief Formats memory volume in @p `bytes` with appropriate units and precision, like @b "1.5 GiB".
 */
struct log_memory_volume_t {

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

/**
 *  @brief Formats a set of CPU core IDs in a compact and readable way, like @b "0-3,5,7,8,10-12".
 */
struct log_core_range_t {

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

/**
 *  @brief NUMA topology logger with compact tree design and color support.
 */
struct log_numa_topology_t {

    /**
     *  @brief Logs NUMA topology in compact tree format with colors.
     *  @param topology The NUMA topology to log
     *  @param colors Color scheme for output formatting
     *  @param output Output file stream (defaults to stdout)
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
                else
                    pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, " "));

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

/**
 *  @brief Logs CPU and memory capabilities summary with compact formatting.
 */
struct log_capabilities_t {

    void operator()(capabilities_t caps, logging_colors_t colors, std::FILE *output = stdout) const noexcept {

        // Line buffer for assembly
        char line_buffer[1024];

        // Helper lambda to flush line buffer
        auto flush_line = [&]() { std::fprintf(output, "%s", line_buffer); };

        // Main header
        std::snprintf(line_buffer, sizeof(line_buffer), "%sSystem Capabilities%s\n", colors.bold_cyan(),
                      colors.reset());
        flush_line();

        // CPU Capabilities row
        std::snprintf(line_buffer, sizeof(line_buffer), "%s├─ %sCPU:%s ", colors.dim(), colors.cyan(), colors.reset());
        std::size_t pos = std::strlen(line_buffer);

        bool first_cpu = true;
        if (caps & capability_x86_pause_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sx86 PAUSE%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_x86_tpause_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sx86 TPAUSE%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_arm64_yield_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sARM64 YIELD%s", first_cpu ? "" : " • ",
                                                          colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_arm64_wfet_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sARM64 WFET%s",
                                                       first_cpu ? "" : " • ", colors.bold_green(), colors.reset()));
            first_cpu = false;
        }
        if (caps & capability_risc5_pause_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sRISC-V PAUSE%s", first_cpu ? "" : " • ",
                                                          colors.bold_green(), colors.reset()));
            first_cpu = false;
        }

        if (first_cpu) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%sNone detected%s", colors.dim(), colors.reset()));
        }

        std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n");
        flush_line();

        // Memory Capabilities row
        std::snprintf(line_buffer, sizeof(line_buffer), "%s└─ %sRAM:%s ", colors.dim(), colors.cyan(), colors.reset());
        pos = std::strlen(line_buffer);

        bool first_mem = true;
        if (caps & capability_place_memory_on_domain_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sNUMA%s",
                                                       first_mem ? "" : " • ", colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }
        if (caps & capability_place_huge_pages_on_domain_k) {
            pos +=
                static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "%s%sHuge Pages%s",
                                                       first_mem ? "" : " • ", colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }
        if (caps & capability_huge_transparent_pages_k) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%s%sTransparent Huge Pages%s", first_mem ? "" : " • ",
                                                          colors.bold_yellow(), colors.reset()));
            first_mem = false;
        }

        if (first_mem) {
            pos += static_cast<std::size_t>(std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos,
                                                          "%sNone detected%s", colors.dim(), colors.reset()));
        }

        std::snprintf(line_buffer + pos, sizeof(line_buffer) - pos, "\n\n");
        flush_line();
    }
};

} // namespace forkunion
} // namespace ashvardanian
