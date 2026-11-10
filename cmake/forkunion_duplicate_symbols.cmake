# Checks the objects of one library, run as `cmake -DOBJECTS=<objects> -P forkunion_duplicate_symbols.cmake`. A text
# symbol several objects define must have the same code in each, since the linker keeps any one of them.
cmake_minimum_required(VERSION 3.28)

find_program(
    nm_
    NAMES llvm-nm nm
    HINTS /opt/homebrew/opt/llvm@22/bin
)
find_program(
    objdump_
    NAMES llvm-objdump objdump
    HINTS /opt/homebrew/opt/llvm@22/bin
)
if (NOT nm_ OR NOT objdump_)
    message(STATUS "Skipping: found no nm or objdump to read the objects with")
    return()
endif ()

foreach (object_ IN LISTS OBJECTS)
    file(
        READ "${object_}" magic_
        LIMIT 4
        HEX
    )
    if (magic_ MATCHES "^(4243c0de|dec0170b)$")
        message(STATUS "Skipping: ${object_} is LTO bitcode, which carries no machine code to compare")
        return()
    endif ()
endforeach ()

execute_process(
    COMMAND ${nm_} --defined-only --extern-only --print-file-name --format=posix ${OBJECTS}
    OUTPUT_VARIABLE all_ COMMAND_ERROR_IS_FATAL ANY
)
string(REGEX MATCHALL "[^\n]+: [^ \n]+ [TW]" all_ "${all_}")

set(duplicated_)
foreach (line_ IN LISTS all_)
    string(REGEX MATCH "^(.+): ([^ ]+) [TW]$" _ "${line_}")
    set(object_ "${CMAKE_MATCH_1}")
    set(symbol_ "${CMAKE_MATCH_2}")
    list(APPEND objects_${symbol_} "${object_}")
    list(LENGTH objects_${symbol_} count_)
    if (count_ EQUAL 2)
        list(APPEND duplicated_ "${symbol_}")
        list(GET objects_${symbol_} 0 first_)
        list(APPEND wanted_${first_} "--disassemble-symbols=${symbol_}")
    endif ()
    if (count_ GREATER 1)
        list(APPEND wanted_${object_} "--disassemble-symbols=${symbol_}")
    endif ()
endforeach ()

execute_process(COMMAND ${objdump_} --help OUTPUT_VARIABLE help_)
set(differing_)
foreach (object_ IN LISTS OBJECTS)
    if (NOT DEFINED wanted_${object_})
        continue()
    endif ()
    # GNU objdump takes one symbol at most, so it disassembles everything and the loop below picks.
    set(only_)
    if (help_ MATCHES "--disassemble-symbols")
        set(only_ ${wanted_${object_}})
    endif ()
    execute_process(
        COMMAND ${objdump_} -d --no-show-raw-insn --no-addresses ${only_} "${object_}"
        OUTPUT_VARIABLE text_ COMMAND_ERROR_IS_FATAL ANY
    )
    # Arm assembly comments start with `;`, which would split the list of symbols below.
    string(REPLACE ";" "#" text_ "${text_}")
    string(REGEX REPLACE " <[^>\n]*>" "" text_ "${text_}")
    string(REGEX REPLACE "0x[0-9a-f]+" "IMM" text_ "${text_}")
    string(REGEX REPLACE "[ \t]+" " " text_ "${text_}")
    string(REGEX REPLACE "(\n (data16 |cs )*(nop[a-z]*|int3)[^\n]*)+\n(\n|$)" "\n\\4" text_ "${text_}")
    string(STRIP "${text_}" text_)
    string(REPLACE "\n\n" ";" text_ "${text_}")
    foreach (chunk_ IN LISTS text_)
        if (NOT chunk_ MATCHES "^\n?<([^>\n]+)>:")
            continue()
        endif ()
        set(symbol_ "${CMAKE_MATCH_1}")
        if (NOT DEFINED objects_${symbol_})
            continue()
        endif ()
        if (NOT DEFINED code_${symbol_}_)
            set(code_${symbol_}_ "${chunk_}")
        elseif (NOT code_${symbol_}_ STREQUAL chunk_ AND NOT symbol_ IN_LIST differing_)
            list(APPEND differing_ "${symbol_}")
        endif ()
    endforeach ()
endforeach ()

set(report_)
foreach (symbol_ IN LISTS differing_)
    list(JOIN objects_${symbol_} "\n    " objects_)
    string(APPEND report_ "\n  ${symbol_} has different code in\n    ${objects_}")
endforeach ()
list(LENGTH OBJECTS objects_)
list(LENGTH duplicated_ duplicates_)
if (report_)
    message(FATAL_ERROR "${objects_} objects, ${duplicates_} text symbols defined more than once:${report_}")
endif ()
message(STATUS "${objects_} objects, ${duplicates_} text symbols defined more than once, all with the same code")
