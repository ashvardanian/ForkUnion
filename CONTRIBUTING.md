# Contributing to ForkUnion

The [Testing & Benchmarking](https://github.com/ashvardanian/ForkUnion/blob/main/README.md#testing--benchmarking) section of the README lists the commands CI runs for C, C++, Rust, Zig and Mojo.
Configuring the CMake build points `core.hooksPath` at `.githooks`, which arms the hooks below.

## Macros

The README lists the public macro families; the internal ones, which end in `_`, are listed here.

| Family                | Form                       | Example                     |
| :-------------------- | :------------------------- | :-------------------------- |
| Architecture fact     | `FORKUNION_ARCH_<ARCH>_`   | `FORKUNION_ARCH_X8664_`     |
| Operating-system fact | `FORKUNION_OS_<OS>_`       | `FORKUNION_OS_LINUX_`       |
| Toolchain fact        | `FORKUNION_HAS_<FEATURE>_` | `FORKUNION_HAS_INLINE_ASM_` |

Architectures are spelled `X8664`, `ARM64`, `RISCV64` and `WASM`.
Only `include/forkunion/types.hpp` reads the operating-system predefines, like `__linux__` or `_WIN32`; compiler predefines, like `__clang__` or `_MSC_VER`, are tested directly.
`defined(...)` appears only in a `!defined` default and in a guard that an `#error` follows.

## Commits

`.githooks/commit-msg` checks every message:

- The title reads `<Verb>: <Summary>`, where the verb is one of `Add`, `Break`, `Chore`, `Docs`, `Fix`, `Improve` or `Make`.
- The summary opens with a capital letter, unless it opens with a code identifier, which keeps its own case.
- The title runs to at most 80 characters, and a body line to at most 100 columns, unless it holds a URL or is indented code.
- The message carries no AI-assistant attribution, since authorship is the human who committed.

Git's merge, `fixup!` and `squash!` titles are exempt from the title rules.
A `Release: vX.Y.Z` title is exempt too, but its commit must stamp every version site, and no other commit may change `VERSION`.
`.githooks/pre-commit` formats and judges the staged lines, and `.githooks/selftest` runs both hooks against their fixtures.
On a pull request, `.github/scripts/check-commits.sh` replays `commit-msg` over every commit and runs `pre-commit` over the whole range.

## Releases

`.github/scripts/mint-release.sh X.Y.Z` cuts a release from a clean checkout of the branch it ships.
It stamps every version site and commits `Release: vX.Y.Z`.
`.github/scripts/check-release.sh` then checks that commit, even in a clone where `core.hooksPath` is unset, before the script tags it and pushes both.
Then it publishes the GitHub release, marking `X.Y.Z-rc.N` candidates as prereleases.
Publishing starts every `release-<binding>.yml` at once, and each runs `check-release.sh` before it builds anything, so each registry trusts only its own binding's file.
