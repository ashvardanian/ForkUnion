# Contributing to ForkUnion

The [Testing & Benchmarking](https://github.com/ashvardanian/ForkUnion/blob/main/README.md#testing--benchmarking) section of the README lists the commands CI runs for C, C++, Rust, Zig and Mojo.
Configuring the CMake build points `core.hooksPath` at `.githooks`, which arms the hooks below.

## Test Suites

### Principles

Every language binding has one suite: `test/main.cpp` for the header, `test/main.c` for the `fu_*` C ABI, the `mod tests` blocks under `rust/forkunion/`, `test/main.mojo` for the Mojo binding, and `zig/` for Zig.
A behaviour that several bindings check carries one stem in all of them: the Rust test name, `test_<stem>` in C, C++ and Mojo.
Tests reach the library through its public surface; the one exception is listed under [Private Surface Waivers](#private-surface-waivers).
Checks never compile out: the C++ and C suites check with `expect_*`, never `assert`, and both open with `#undef NDEBUG // ! A test's oracle must stay live in every build`.
The benchmarks never include `test/`, and neither tree uses a test or benchmark framework.
The harness owns stdout; a case writes its diagnostics to stderr.
Settings are read once at start, and a value that does not parse prints one line and exits with 1.
Skips are honest: a case that cannot run says why, and a run whose filter selects nothing fails.

The C++ suite has one mutable global, `test_run`, the run in progress.
Checks fire on worker threads and inside helpers as well as in a case's own body, so the run counts failed checks in one place and `run_test_case` diffs that count around each case.
After a crash, `p test_run` in a core dump names the case in progress and the last failed check, and the crash handler prints the case too.

### Running

```bash
cmake --preset debug                    # Clang, with the tests and benchmarks
cmake --build --preset debug
ctest --preset debug                    # or `ctest --test-dir build_debug --output-on-failure`
cargo test                              # Rust, as CI runs it; add `--features place-memory-on-domain` or
                                        # `--no-default-features --features portable,std` for the other feature sets
pixi run test                           # Mojo
zig build test --summary all            # Zig
```

The C++ suite builds twice, as `forkunion_test_cpp20` and `forkunion_test_cpp23`, and the C suite as `forkunion_test_c11`, plus a GCC nested-functions or Clang blocks variant where the compiler offers one.
Every case has a label that names the code under test, spelled as the code spells it:

- a case over a runtime or template choice reads as a call with that choice, as `test_for_n_visits_each_task_once_and_again<make_distributed_pool_t>`;
- C labels are the test function plus the capability mask the battery ran it under, as `test_spawn_rejects_zero_threads(fu_capability_arm64_yield_k | fu_capability_os_threads_k)`;
- the epoch-wrap stress cases carry their arguments, as `test_mixed_dispatch_survives_epoch_wraps<fu8_t>(7, max_stress_threads, 1)`, and a build with `FORKUNION_TEST_SKIP_STRESS` skips them.

The filter is a regular expression:

```bash
FORKUNION_FILTER='<make_distributed_pool_t>$' build_debug/forkunion_test_cpp20 # every case on the distributed pool
FORKUNION_FILTER='^test_for_n_dynamic' build_debug/forkunion_test_cpp20        # one subject on every pool
FORKUNION_FILTER=test_forks_from_thread_zero build_debug/forkunion_test_c11
```

Each case prints one verdict line, and the run ends with a summary:

```text
- Seed: 42, scale: 1, filter: none
- test_spawn_succeeds_at_full_width<> ... ok (0.05 s)
- test_for_n_dynamic_covers_every_regime_across_domains ... skipped: needs 2+ compute domains
103 passed, 0 failed, 1 skipped, seed 42
```

A failed check prints where it failed, with both operands for `expect_equal`, and the case goes on, so one run lists every mismatch it reaches.
A case with a failed check prints a `rerun:` line naming the binary, the seed, and a filter that selects only that case:

```text
  check failed at /path/to/ForkUnion/test/main.cpp:1004
    expected 10000, got 9999
- test_for_n_visits_each_task_once_and_again<make_colocated_pool_t> ... FAILED: 1 failed check(s)
  rerun: build_debug/forkunion_test_cpp20 --seed 42 --filter '^test_for_n_visits_each_task_once_and_again<make_colocated_pool_t>$'
```

The run exits with 1 when a case failed or none ran, with 3 when `FORKUNION_NO_SKIPS` is set and a case skipped, and with 0 otherwise.
`FORKUNION_TRAP_ON_FAILURE=1` raises `SIGTRAP` at the first failed check, which stops a debugger there, or dumps a core without one:

```bash
FORKUNION_TRAP_ON_FAILURE=1 FORKUNION_FILTER='^test_forks_' gdb -ex run --args build_debug/forkunion_test_cpp20
```

A crash prints `*** SIGSEGV in case '<label>' - backtrace follows ***`, then a backtrace where the C library offers one.

### Environment

The C++ and C suites read these settings, each from a flag or else its variable:

| Variable                    | Flag                | Default | Meaning                                                                  |
| :-------------------------- | :------------------ | :------ | :----------------------------------------------------------------------- |
| `FORKUNION_SEED`            | `--seed`            | `42`    | Seed of the run, or `random`; it names the run in every `rerun:` line    |
| `FORKUNION_SCALE`           | `--scale`           | `1`     | Multiplier for the repetition counts, like `0.1` or `10`                 |
| `FORKUNION_FILTER`          | `--filter`          | unset   | ECMAScript regex over the labels, else a substring; C matches substrings |
| `FORKUNION_TRAP_ON_FAILURE` | `--trap-on-failure` | unset   | C++ only: raises `SIGTRAP` at the first failed check                     |
| `FORKUNION_NO_SKIPS`        | `--no-skips`        | unset   | Exits with 3 when any case skipped                                       |

A switch is off when unset, empty, `0`, `false`, `off` or `no`, and on otherwise.
No case draws random inputs yet, so the seed reproduces nothing beyond itself; a case that draws takes `split_mix_generator_t {case_stream_key()}`, keyed by its own label, so a filtered run draws what an unfiltered one did.
`FORKUNION_SCALE` scales the generation and churn cases in C++ and the refusal rounds in C, never the counts a case needs to reach a boundary, like the epochs that wrap a `uint8` clock.
The Rust and Mojo suites filter through `cargo test <name>` and their own runner, and read none of these.

### Writing a Test

A C++ case is a `static` function named `test_<behaviour>`, registered in `main` as `run_test_case("<area>.<case>", test_<behaviour>, arguments...)`.
A case that runs on every pool is a template over its pool maker, registered once per pool with the `.flat`, `.colocated` and `.distributed` variant.
It checks with `expect_true(held)` or `expect_equal(actual, wanted)`, each of which counts a failed check, lets the case go on, and returns whether it held.
Where the next step would index or dereference past what it checked, the case stops on the answer, as in `if (!expect_true(fu::succeeded(pool.spawn(threads)))) return;`.
A case that cannot run returns `case_result_t::skipped(reason)`, a case that ran returns nothing or `{}`, and a reason takes one of four forms:

- `needs <resource>`, as `needs 2+ compute domains`;
- `built without <SWITCH>`, as `built without FORKUNION_WITH_OS_THREADS`;
- `<facility> is refused on this machine`, as `thread affinity is refused on this machine`;
- `no <device>`.

A wait on another thread goes through `spin_until(predicate)`, whose `wait_limit_k` bounds a hang without timing anything, and a repetition count goes through `scaled_iterations(baseline)`.
A reusable checking helper is named `check_<what>`, a builder `make_<what>`, and a trailing underscore marks the harness's own helpers alone.
A C case takes the capability mask, checks with `expect_(condition)`, or `expect_at(held, __FILE__, __LINE__, "message")` for a message, and joins the `run_battery` table.
A Rust test is named as a behaviour sentence, skips with `return skipped("reason")` from `rust/testing.rs`, and carries `#[cfg_attr(miri, ignore)]` when it reaches the C core, which Miri cannot call.
A Mojo test is `test_<stem>`, registered in `main` with `suite.test[...]()`, and skips with `_skipped("reason")`.

The pinned regions, `Shared Settings` and `Shared Run` in `test/main.cpp`, `Shared Settings` in `bench/harness.hpp`, `Shared C Checks` in `test/main.c`, and `Shared Testing` in `rust/testing.rs` and `test/main.mojo`, are identical across the projects this one shares a harness with.
Edit them nowhere but at their source; a formatter may lay them out its own way.

### Benchmarks

`bench/nbody.*` and `bench/propagation.*` share a harness per language: `bench/harness.hpp`, `harness.rs`, `harness.zig` and `harness.mojo`.
Every port reads the same variables once at start; the C++ ones also take each as a flag, which overrides the variable:

| Variable                            | Flag                        | Default                   | Meaning                              |
| :---------------------------------- | :-------------------------- | :------------------------ | :----------------------------------- |
| `FORKUNION_SEED`                    | `--seed`                    | `42`                      | Seed of every draw, or `random`      |
| `FORKUNION_WARMUP`                  | `--warmup`                  | `1s`                      | Untimed run before timing            |
| `FORKUNION_TIME_LIMIT`              | `--time-limit`              | `10s`                     | Timed window, like `500ms`           |
| `FORKUNION_THREADS`                 | `--threads`                 | all cores                 | Threads; `0` also means all cores    |
| `FORKUNION_BACKEND`                 | `--backend`                 | `forkunion_static_shared` | Backend, per benchmark header        |
| `FORKUNION_NBODY_COUNT`             | `--nbody-count`             | the thread count          | Bodies to simulate                   |
| `FORKUNION_PROPAGATION_SCALE`       | `--propagation-scale`       | `14`                      | `2^scale` vertices per community     |
| `FORKUNION_PROPAGATION_COMMUNITIES` | `--propagation-communities` | `64`                      | Communities strung on the ring       |
| `FORKUNION_PROPAGATION_EDGE_FACTOR` | `--propagation-edge-factor` | `16`                      | Raw edges per vertex                 |
| `FORKUNION_PROPAGATION_CHECK`       | `--propagation-check`       | `0`                       | `1` also converges serially to check |

Leaving a setting unset or empty keeps its default, and a value that does not parse prints one line and exits with status 1.

```bash
FORKUNION_NBODY_COUNT=512 build_release/forkunion_nbody --backend forkunion_dynamic_shared --time-limit 30s
FORKUNION_PROPAGATION_CHECK=1 build_release/forkunion_propagation
```

### Devices

ForkUnion has no device backends, so every suite runs on the host CPU alone.

### Private Surface Waivers

`test/main.cpp` calls `fu::derive_memory_levels_` in `fabric.ranks_bandwidth_before_latency`, because ranking a synthetic edge log is the only way to reach every tier boundary; the public `measured_fabric_t::harvest` feeds it only from the machine it runs on.

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
