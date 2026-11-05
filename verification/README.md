# Verification

Model checking for the words ForkUnion's atomics touch, and the C++ memory model the models run under.
Two tools, both open source and neither on the JVM: [Spin](https://spinroot.com) for the protocol layer, [GenMC](https://github.com/MPI-SWS/genmc) for the C++ source under SC, TSO, RC11 and IMM.
`./check.sh` runs every `@verify` line here and compares each verdict with the expected one.

- `weak_memory.pml` is the C++ memory model as views, for Promela: relaxed, acquire, release, `acq_rel`, both fences, release sequences, and the far model's posted adds.
- `weak_memory_litmus/` is its calibration, one classic shape per file, each asserting the outcome RC11 forbids and expecting exactly RC11's verdicts.
- `monitor_wait.pml` is a wait that can miss its wake-up: a waiter arms, re-checks the word, and blocks until a writer's `wake`, so a lost notify reads as `stuck`.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `flat_pool/`, `colocated_pool.pml`, `distributed_pool/`, `for_n_dynamic.pml`, `spin_mutex.pml` and `standard_atomic_ref.cpp` check the pools and the portable atomics, each file opening with what it checks.

## Writing models

Every model opens the same way: the file docblock, the include, then the names.
The comments follow the C++ headers' rules, and the pre-commit hook checks `*.pml` with the same Doxygen checks.
Every `inline` and `proctype`, and every documented group of `#define`s, carries a `/** */` docblock directly above it whose first sentence is its summary.
A note over a whole `#if` branch is a plain `/* */` above the `#if`, and `//` is kept for notes inside a body.
Inside a block, a single name reads `@c name`, a parameter of the inline below reads `@p name`, and a span of several tokens stays in backticks.
Code is cited by symbol and file, as `flat_pool::unsafe_join` in `flat.hpp`, never by line number, which drifts with every edit above it.
Everything is lowercase snake case; pan's own `-DSAFETY`, `-DCOLLAPSE` and `-DVECTORSZ` in the runner are the only capitals.
A word carries the name of the C++ member it stands for, without the trailing underscore, or a small accessor like `row_lock(id)` when several rows share a shape.
Field extractors read `<field>_of(word)`, transforms are past participles like `stepped(word)`, per-site orders are `<site>_order`, values made from an ID are `<state>(id)`.

A component is one mechanism of the code under test.
It is a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second: a `protocol.pml` beside one file per scenario, and a `client.cpp` when GenMC checks the same words.
The protocol owns the word map, the values, the knobs and the inlines; words never overlap, and the ones most scenarios touch come first.
A scenario is one cast of processes, and opens with its docblock, its `@verify` lines and its constants: `thread_count`, `location_count`, one past the highest word it touches, and `history_depth`, the most writes one word receives, the initial one included.
Then it includes its protocol, declares its roles as `proctype <role>(byte t)`, and starts them from one `init { atomic { run <role>(<thread>); … } }`, so every thread index is fixed and no two identical processes race for one.
There is no conditional compilation in protocols, scenarios or clients.
The preprocessor names words, values and constants, gives each knob its default under `#ifndef`, and range-checks an enumerated knob with `#error`; everything that differs between memory models or knob values is a plain Promela expression, `if :: memory == sc -> … :: else fi` or `if :: back_link_fence -> fence_release(t) :: else fi`.
Write such a guard as a bare `if`: statement merging folds the constant guard into its neighbour, while an `atomic` around it makes a step of its own.
An inline may declare its own locals, so a role is often one line, unless that adds states: each declaration is a step of its own, and inside a loop it resets on every pass, so where a scenario's states grow the role declares them and passes them in.
An inline's local never shares a name with its caller's.
A wait on a word another thread moves goes through `monitor_wait.pml`, never a guard on the newest value, which cannot miss a wake-up.

Every file opens with one docblock:

- `@file`, `@author`, `@date` of the file's first commit, and `@brief`;
- one context paragraph of five lines or fewer;
- one-line property bullets;
- an `@budget` line, only when it differs from 1024;
- `@verify` lines with short reasons.

## Knobs and verdicts

A knob is a choice the code under test makes, named for the mechanism and defaulting to the code: `watermark_order=order_relaxed`, `back_link_fence=false`, `commits=2`, `waiting=parking`.
An order knob ends in `_order`; a boolean names the step the code takes.
A client spells the same knobs as `bool` and `std::memory_order` fields of a `knobs_t`, one `constexpr knobs_t <variant>_k` per variant, read through a `template <knobs_t const &knobs_>` with `if constexpr`; GenMC cannot load a class passed by value as a template argument.
Each variant is an `extern "C"` entry named `<scenario>` or `<scenario>_<knob>_<value>`, and a flat client's scenario is its component.

Every expected verdict is a `@verify` line in the docblock of the file it checks:

```
@verify <pass|fail|stuck> <memory>[,<memory>...] [<knob>=<value> ...][: <finding>]   in a scenario
@verify <pass|fail|stuck> <memory>[,<memory>...] <entry>[: <finding>]                in a client
```

Spin runs `sc`, `rc11` and `far`; GenMC runs `sc`, `tso`, `rc11` and `imm`, and IMM reads the client as compiled code on x86, Arm and POWER.
`pass` is a search that completed with no error, `fail` a violated assertion, `stuck` an invalid end state in Spin or a liveness violation in GenMC.
A search cut short by the depth limit, a knob defined twice, an index or a write past the scenario's shape, or a GenMC pass that explored nothing is `broken`.
Every scenario has a `pass` line, and every knob appears on some line; `verify_all` checks the second both ways.
A `fail` or `stuck` line's finding states the mechanism and its counterexample in the present tense: what the code does, and what happens without it.
Adding a scenario is one file, a knob is one `#ifndef` and its lines, a new set of words is a new component, and a finding is the line that replays it; nothing ever edits `check.sh`.

## Memory models

- `-Dmemory=sc`: one copy of every location, every access one step.
  The protocol layer, and where logic bugs are found first.
- `-Dmemory=rc11`, the default: every location is a history of writes, every thread carries a view over the histories, a release write stamps the writer's view onto the write and an acquire read merges it.
- `-Dmemory=far`: the views, plus a relaxed no-return add is posted rather than performed.

## Two include roots

Every `#include` is an angle-bracket path from one of two roots, ForkUnion's `verification/` and the repo's own.
Harness files are named bare, as `<weak_memory.pml>`, `<monitor_wait.pml>` and `<genmc.hpp>`; a repo's models from its root, as `<flat_pool/protocol.pml>`; library headers as `<forkunion/atomics.hpp>`.
No include climbs with `..` or uses quotes, and the runner refuses a repo file named like a harness file, which would shadow it.

## Memory budget

- Each model's `@budget <MB>` caps its search, 1024 by default, and a search past it reads as `broken`.
- `-w` is derived so the hash table takes half the budget, capped at `-w27`.
- Verdicts pack into `VERIFY_MEMORY_MEGABYTES`, three quarters of available memory by default.
- `@budget` is the next power of two at or above three times the run's states tally, which is its tally less the hash table the budget gave it.
  The states then fill at most two thirds of the half the table leaves.
- Up to 4096 MB a model runs in CI; from 4096 to 8192 its header justifies the budget in one line; above 8192 it runs only with `VERIFY_BACKGROUND=1`.

## Running

```sh
./check.sh                                   # Spin only, GenMC skipped when absent
./check.sh flat_pool/moods.pml               # one scenario
GENMC=~/genmc/build/bin/genmc ./check.sh     # both
```

Another repo's `verification/check.sh` sources this runner from its `forkunion` submodule or from `FORKUNION_SOURCE`.
The far model runs only on the litmus shapes, since no pool path posts a relaxed no-return add.
`GENMC_CLANG` names the compiler GenMC was built with, `clang++` by default; `GENMC_SECONDS` caps one client, 300 by default; `GENMC_UNROLL` gives every loop that many turns, 3 by default, since GenMC treats a weak compare-exchange as one that may fail spuriously and a read-first retry loop never ends for it.
Every verdict runs in its own directory, at most `VERIFY_JOBS` at a time, and the lines print in the order they were queued once the last one lands.

GenMC ships a freestanding C library whose headers shadow the platform's, `atomic` without `std::atomic_ref` among them, and puts its include directory behind the caller's.
The clients need the real standard library, so `genmc_ready` writes a directory of forwarders for the shadowed names and puts it first.
The clients spawn through `__VERIFIER_thread_create`: the platform's `pthread_create` is not intercepted, and `std::thread` rides on it, which is why `flat_pool/client.cpp` spells the pool's words rather than running the pool.

## When a check fails

A `WRONG` line names the model, the defines it was built with, the result expected and the result seen.

- A `pass` model that failed means the design admits the outcome an assertion forbids.
  Read the counterexample: `spin -t -p flat_pool/plain.pml` replays the trail Spin left beside `pan`, one step per line, and the last steps before the assertion are the interleaving to fix.
  Fix the design or the model, and say in the commit which one was wrong.
- A `fail` knob that passed means the model no longer exercises what the knob breaks, usually because a parameter shrank below the shape the bug needs.
  This is the worse failure of the two, since it silently turns the model into a check of nothing.
- A `broken` result is Spin or the compiler refusing the model, and the first lines of the log printed underneath say why.

Never relax an assertion to make a run green.
An assertion that is wrong is a design decision to revisit in its own commit, with the reason in the message.

## What the models found

The pool's three atomics are sound, and the models prove the behaviours composed on them; what they found sits at the edges.
Where a model replays a finding, its `@verify` line links it.

- A dispatch waking a parked pool restores the workers' scheduling class to `SCHED_OTHER`, the class they started on, since `SCHED_FIFO | SCHED_RR` is `SCHED_BATCH` on Linux.
- Parked workers wait on the mood word with no cap, so a store to it wakes a monitor, while a kernel wait needs an explicit wake, which `nap` owes a `sleep` before it, and the dispatch's wake-up is a release, or a worker leaving its startup wait on it reads a stale handle: [`flat_pool/moods.pml`](flat_pool/moods.pml).
- `terminate` is documented as callable only with no task running and the last dispatch joined, which it asserts.
- A dispatch claims the pool by exchanging the contributor countdown from zero with an acquire, writes its fork only after the claim, and waits out the previous generation's completion step, or a refused caller tears the running fork, or a token comes out even: [`flat_pool/contenders.pml`](flat_pool/contenders.pml).
- `distributed_pool` claims every colocation before it publishes on any, or a colocation that refuses leaves the others running half a dispatch: [`distributed_pool/contenders.pml`](distributed_pool/contenders.pml).
- The pool rewinds the shared slice cursors of a `for_n_dynamic` only once its claim admitted the dispatch, or a refused nested call rewinds them under the running one, which runs some tasks twice: [`for_n_dynamic.pml`](for_n_dynamic.pml).
- Every monitored wait re-checks its word after arming, or a worker sleeps through the dispatch or the completion that already came: [`flat_pool/plain.pml`](flat_pool/plain.pml), [`colocated_pool.pml`](colocated_pool.pml).
- The C shim's `fu_pool_unsafe_join` clears its callback slot only for the generation it published, or a stale join empties the live one's slot under a running worker: [`flat_pool/stale_join.pml`](flat_pool/stale_join.pml).
- The epoch's dispatch release, the workers' pickup acquire, the last contributor's completion release and the join's acquire each carry the fork and its results, and the countdown's decrements acquire as well as release, or the last contributor reads a stale result, under RC11 and under IMM: [`flat_pool/`](flat_pool/).
- `spin_mutex`'s `lock` and `lock_if_free` acquire and its `unlock` releases, or the next holder reads the last one's writes stale: [`spin_mutex.pml`](spin_mutex.pml).
- The loops behind `fetch_add_if_at_most` and `fetch_max` acquire even when they refuse, or a refused caller reads the winner's data stale, under RC11 and under IMM: [`standard_atomic_ref.cpp`](standard_atomic_ref.cpp).
- The workers re-check the mood under a capped wait, which bounds a `nap`, `sleep` or `terminate` notice to one timeout; `terminate` re-releases the epoch after die, which wakes the workers asleep on it but not one arming after it: [`flat_pool/moods.pml`](flat_pool/moods.pml).
- The epoch's width aliases at the debug widths, as the header prices: at a modulus of 2 a stale token names the live generation: [`flat_pool/polling.pml`](flat_pool/polling.pml).
- `invoke_for_n_dynamic::rewind` writes the cursors relaxed, since the dispatch's release on the epoch carries every cursor and end to every worker: [`for_n_dynamic.pml`](for_n_dynamic.pml).
- Both epoch steps are plain release stores: the claim leaves the dispatcher the epoch's only writer until its contributors count down, and the next claimer waits for the last contributor's even epoch before it writes: [`flat_pool/`](flat_pool/), [`distributed_pool/`](distributed_pool/).
- `unsafe_join` runs the caller's slice on its own colocation before it waits for any other, or the join and a worker waiting on the caller's slice wait on each other forever: [`distributed_pool/overlap.pml`](distributed_pool/overlap.pml).
- A `broadcast_join` kept alive across a `terminate` and a `spawn` would alias after one re-spawn rather than after `2^bits` epochs, since `terminate` resets the epoch; it asserts every dispatch joined, so nothing outlives it by contract.
