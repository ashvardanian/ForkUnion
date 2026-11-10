# Sanitizers, and the warning flags ForkUnion's own targets take through `forkunion_warnings`.

# Sanitizers instrument every target this directory builds, never the probes, and never a parent project's own targets.
if (FORKUNION_BUILD_SANITIZERS STREQUAL "address" AND MSVC)
    add_compile_options(/fsanitize=address)
elseif (FORKUNION_BUILD_SANITIZERS STREQUAL "address")
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all -UNDEBUG -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
elseif (FORKUNION_BUILD_SANITIZERS STREQUAL "thread" AND NOT MSVC)
    add_compile_options(-fsanitize=thread -UNDEBUG -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
elseif (NOT FORKUNION_BUILD_SANITIZERS STREQUAL "off")
    message(FATAL_ERROR "FORKUNION_BUILD_SANITIZERS is off, address or thread, the last with Clang or GCC, "
                        "not '${FORKUNION_BUILD_SANITIZERS}'"
    )
endif ()

# Strictest possible warnings for ForkUnion's own libraries, tests and benchmarks, never passed on to consumers.
add_library(forkunion_warnings INTERFACE)
if (FORKUNION_WERROR)
    target_compile_options(
        forkunion_warnings INTERFACE $<$<CXX_COMPILER_ID:GNU,Clang>:-Werror> $<$<CXX_COMPILER_ID:MSVC>:/WX>
    )
endif ()
target_compile_options(
    forkunion_warnings
    INTERFACE # GCC/Clang: Maximum warnings + security hardening
              $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall
              -Wextra
              -Wpedantic
              -Wundef
              -Wconversion
              -Wcast-qual
              -Wcast-align
              -Wunused
              -Wno-unused-parameter
              -Wno-unknown-pragmas
              -Wno-sign-conversion
              -Wno-unused-function
              -Wshadow
              -Wno-null-dereference
              -Wdouble-promotion>
              # C++ only: GCC rejects these on the C tests and on a consumer's C sources
              $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wnon-virtual-dtor
              -Wold-style-cast
              -Woverloaded-virtual
              -Wsign-promo>
              $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wuseless-cast>
              $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release,RelWithDebInfo>>:-Wno-unused-variable>
              # Additional GCC-specific warnings
              $<$<CXX_COMPILER_ID:GNU>:-Wmisleading-indentation
              -Wduplicated-cond
              -Wduplicated-branches
              -Wlogical-op
              -Wno-null-dereference>
              # Additional Clang-specific warnings
              $<$<CXX_COMPILER_ID:Clang>:-Wmost
              -Wthread-safety
              -Wthread-safety-negative>
              $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wdocumentation>
              # MSVC: Maximum warnings
              $<$<CXX_COMPILER_ID:MSVC>:/W4
              /permissive-
              /w14242
              /w14254
              /w14263
              /w14265
              /w14287
              /we4289
              /w14296
              /w14311
              /w14545
              /w14546
              /w14547
              /w14549
              /w14555
              /w14619
              /w14640
              /w14826
              /w14905
              /w14906
              /w14928>
              $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release,RelWithDebInfo>>:/wd4101>
)
