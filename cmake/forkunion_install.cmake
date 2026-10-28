# Install rules, the CMake package files, and the `.pc`.

# Install rules. Default ON only when ForkUnion is the top-level project: an `add_subdirectory` consumer installing
# itself would otherwise drop ForkUnion's headers, libraries, and CMake package into *its* prefix, colliding with a real
# `forkunion` package. That is the documented path - `FetchContent_MakeAvailable` adds the subdirectory plainly, without
# the `EXCLUDE_FROM_ALL` that would suppress these on newer CMake. A parent that does want ForkUnion packaged alongside
# it sets `FORKUNION_INSTALL` first; `CMP0077` above lets that normal variable win.
option(FORKUNION_INSTALL "Install ForkUnion's headers, libraries, and CMake package" ${PROJECT_IS_TOP_LEVEL})

if (FORKUNION_INSTALL)
    include(GNUInstallDirs)
    install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # No `INCLUDES DESTINATION` here: `target_include_directories` above already declares the same
    # `$<INSTALL_INTERFACE:include>`, and naming it twice exports the path twice.
    install(TARGETS forkunion_header EXPORT forkunionTargets)
    install(
        TARGETS ${forkunion_libraries_}
        EXPORT forkunionTargets # same export set
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # .so / .dylib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # .a
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # .dll on Windows
    )

    # Export config files for find_package
    include(CMakePackageConfigHelpers)
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        cmake/forkunionConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
    )

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/forkunionConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
    )

    install(
        EXPORT forkunionTargets
        NAMESPACE forkunion::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/forkunion
    )
endif ()

# A `.pc` for the consumers CMake never reaches. Must follow the two probes above: `configure_file` substitutes now,
# unlike `install(EXPORT)`, which writes at generate time and so tolerates a link interface declared later. Spelling the
# prefix against `${pcfiledir}` keeps it correct after `cmake --install --prefix` moves the tree. Under the same option
# as the rules above, which is also what brought `GNUInstallDirs` in - without it these paths resolve empty.
if (FORKUNION_INSTALL)
    file(RELATIVE_PATH forkunion_pc_relative_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
    string(REGEX REPLACE "/+$" "" forkunion_pc_trimmed_prefix_ "${forkunion_pc_relative_prefix_}")
    set(forkunion_pc_prefix_ "\${pcfiledir}/${forkunion_pc_trimmed_prefix_}")

    # `GNUInstallDirs` returns a relative path unless a distro forced an absolute one; only the former may follow a
    # prefix.
    if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
        set(forkunion_pc_libdir_ "${CMAKE_INSTALL_LIBDIR}")
    else ()
        set(forkunion_pc_libdir_ "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
    endif ()
    if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(forkunion_pc_includedir_ "${CMAKE_INSTALL_INCLUDEDIR}")
    else ()
        set(forkunion_pc_includedir_ "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
    endif ()

    # A C caller linking the archive has to name the C++ runtime itself, so read it off the compiler rather than the
    # platform. These go in `Libs.private`, emitted only for `--static`; `Requires.private` would name pkg-config
    # modules, and none of the three ship one.
    set(forkunion_pc_runtime_libraries_ "")
    foreach (implicit_library_ IN LISTS CMAKE_CXX_IMPLICIT_LINK_LIBRARIES)
        if (implicit_library_ MATCHES "^(stdc\\+\\+|c\\+\\+)$")
            list(APPEND forkunion_pc_runtime_libraries_ "-l${implicit_library_}")
        endif ()
    endforeach ()
    if (forkunion_atomics_are_builtin_)
        set(forkunion_pc_atomic_library_ "")
    else ()
        set(forkunion_pc_atomic_library_ "-latomic")
    endif ()
    string(JOIN " " forkunion_pc_private_ ${forkunion_pc_runtime_libraries_} ${CMAKE_THREAD_LIBS_INIT}
           ${forkunion_pc_atomic_library_}
    )

    configure_file(cmake/forkunion.pc.in "${CMAKE_CURRENT_BINARY_DIR}/forkunion.pc" @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/forkunion.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif ()
