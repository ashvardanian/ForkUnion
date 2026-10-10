# Verification

Model checking for the words ForkUnion's atomics touch, and the C++ memory model the models run under.
Two tools, both open source and neither on the JVM: [Spin](https://spinroot.com) for the protocol layer, [GenMC](https://github.com/MPI-SWS/genmc) for the C++ source under SC, TSO, RC11 and IMM.
`./check.sh` runs every `@verify` line here and compares each verdict with the expected one.

- `weak_memory.pml` is the C++ memory model as views, for Promela: relaxed, acquire, release, `acq_rel`, both fences, release sequences, and the far model's posted adds.
- `weak_memory_litmus/` is its calibration, one classic shape per file, each asserting the outcome RC11 forbids and expecting exactly RC11's verdicts.
- `check.sh` is the runner, and `genmc.hpp` the threads and the assertion a client takes from GenMC.
- `flat_pool/`, `for_n_dynamic.pml`, `distributed_pool/` and `standard_atomic_ref.cpp` check the pools and the portable atomics, each file opening with what it checks.

## Conventions

Every model opens the same way: the file docblock, the include, then the names.
The comments follow the C++ headers' rules, and the pre-commit hook checks `*.pml` with the same Doxygen checks.
The file docblock opens with `@file`, `@author`, `@date` and a `@brief` of at most three lines, then the body, where each invariant and each scenario is a paragraph opening with its subject.
Every `inline` and `proctype`, and every documented group of `#define`s, carries a `/** */` docblock directly above it whose first sentence is its summary.
A note over a whole `#if` branch is a plain `/* */` above the `#if`, and `//` is kept for notes inside a body.
Inside a block, a single name reads `@c name`, a parameter of the inline below reads `@p name`, and a span of several tokens stays in backticks.
Code is cited by symbol and file, as `flat_pool::unsafe_join` in `flat.hpp`, never by line number, which drifts with every edit above it.
Everything is lowercase snake case; pan's own `-DSAFETY`, `-DCOLLAPSE` and `-DVECTORSZ` in the runner are the only capitals.
A word carries the name of the C++ member it stands for, without the trailing underscore, or a small accessor like `row_lock(id)` when several rows share a shape.
Field extractors read `<field>_of(word)`, transforms are past participles like `stepped(word)`, per-site orders are `<site>_order`, values made from an ID are `<state>(id)`.

## Writing a model

A component is one mechanism of the code under test.
It is a flat `<name>.pml` while it has one scenario, and a directory `<name>/` once it has a second: a `protocol.pml` beside one file per scenario, and a `client.cpp` when GenMC checks the same words.
The protocol owns the word map, the values, the knobs and the inlines; words never overlap, and the ones most scenarios touch come first.
Scenarios whose words cannot share one map are separate components, and a protocol several components need sits flat beside them, taking words as arguments.

A scenario is one cast of processes, and opens with its docblock, its `@verify` lines and its constants: `thread_count`, `location_count`, one past the highest word it touches, and `history_depth`, the most writes one word receives, the initial one included.
Then it includes its protocol, declares its roles as `proctype <role>(byte t)`, and starts them from one `init { atomic { run <role>(<thread>); … } }`, so every thread index is fixed and no two identical processes race for one.
The depth is the exact minimum every one of the scenario's `rc11` and `far` lines accepts, failing ones included, since a mutant may write more before it reaches its counterexample: `sc` keeps no history, and a depth too small reports `history_full` and reads as `broken`.

There is no conditional compilation in protocols, scenarios or clients.
The preprocessor names words, values and constants, gives each knob its default under `#ifndef`, and range-checks an enumerated knob with `#error`; everything that differs between memory models or knob values is a plain Promela expression, `if :: memory == sc -> … :: else fi` or `if :: back_link_fence -> fence_release(t) :: else fi`.
Write such a guard as a bare `if`: statement merging folds the constant guard into its neighbour, while an `atomic` around it makes a step of its own.
An inline may declare its own locals, so a role is often one line, unless that adds states: each declaration is a step of its own, and inside a loop it resets on every pass, so where a scenario's states grow the role declares them and passes them in.
An inline's local never shares a name with its caller's.

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

`./check.sh` runs every `@verify` line here and one directory down, and takes files to run only those.
Adding a scenario is one file, a knob is one `#ifndef` and its lines, a new set of words is a new component, and a finding is the line that replays it; nothing ever edits `check.sh`.

## Three memory models, one interface

Every model passes its own thread index to `load`, `store`, `read_modify_write`, `read_modify_write_if`, `compare_exchange`, `add_no_return`, `fence_acquire` and `fence_release`, and `-Dmemory=` picks the memory model at `spin -a` time:

- `-Dmemory=sc`: one copy of every location, every access one step.
  The protocol layer, and where logic bugs are found first.
- `-Dmemory=rc11`, the default: every location is a history of writes, every thread carries a view over the histories, a release write stamps the writer's view onto the write and an acquire read merges it.
  This is the view semantics of Kaiser, Dang, Dreyer, Lahav and Vafeiadis without promises, which is RC11's release-acquire-relaxed fragment: load buffering is forbidden, as in RC11.
- `-Dmemory=far`: the views, plus a relaxed no-return add is posted rather than performed.
  It lands at some later step in the `far_cache` process, and until then no release by the posting thread carries it; a same-address access by the poster lands it first.
  This is RAO-INT as Intel documents it: `AADD` and kin are "implemented using the weakly-ordered memory consistency model of write combining (WC) memory type", and "a fencing operation implemented with LFENCE, SFENCE, or MFENCE instruction should be used in conjunction with AADD if a stronger ordering is required".
  Nothing else is promised to order them, and a C++ release fence compiles to no instruction on x86, so a relaxed no-return add followed by a release fence or a release store is posted in the model exactly as it is in silicon.
  The ForkUnion reference maps only relaxed no-return operations onto them, and a release order on the same call is a lock-prefixed instruction, which is how the index sites that need the order now spell it.

The calibration in `weak_memory_litmus/` pins the model to RC11 on message passing with and without releases and fences, release sequences continued by a read-modify-write and broken by a store, coherence, load buffering, and the far shape: a relaxed no-return add before a release store, carried in C++ and posted under far memory.

## What the models found

The pool's three atomics are sound, and the models prove the behaviours composed on them; what they found sits at the edges.
Each finding is the `@verify` line that replays it, where the full counterexample is written out.

- A dispatch waking a chilled pool restores the workers' scheduling class to `SCHED_OTHER`, the class they started on, since `SCHED_FIFO | SCHED_RR` is `SCHED_BATCH` on Linux.
- The same wake-up exchanges the mood strongly, since a spurious failure of a weak one leaves a colocated worker in its startup wait forever: [`flat_pool/moods.pml`](flat_pool/moods.pml).
- `terminate` is documented as callable only with no task running and the last dispatch joined, which it asserts.
- The C shim's `fu_pool_unsafe_join` clears its callback slot only for the generation it published, or a stale join empties the live one's slot under a running worker: [`flat_pool/stale_join.pml`](flat_pool/stale_join.pml).
- The countdown's decrements acquire as well as release, or the last contributor reads a stale result, under RC11 and under IMM: [`flat_pool/`](flat_pool/).
- The workers re-check the mood under a capped wait, which bounds a `sleep` or a `terminate` notice to one timeout: [`flat_pool/moods.pml`](flat_pool/moods.pml).
- The epoch's width aliases at the debug widths, as the header prices: at a modulus of 2 a stale token names the live generation: [`flat_pool/polling.pml`](flat_pool/polling.pml).
- The release on each cursor in `invoke_for_n_dynamic::reset_slices_` is redundant, since the dispatch's release on the epoch carries every cursor and end to every worker, and the release stays: [`for_n_dynamic.pml`](for_n_dynamic.pml).
- `unsafe_join` runs the caller's slice on its own colocation before it waits for any other, or the join and a worker waiting on the caller's slice wait on each other forever: [`distributed_pool/overlap.pml`](distributed_pool/overlap.pml).

A `broadcast_join` kept alive across a `terminate` and a `spawn` would alias after one re-spawn rather than after `2^bits` epochs, since `terminate` resets the epoch; it asserts every dispatch joined, so nothing outlives it by contract.

## Running

```sh
./check.sh                                   # Spin only, GenMC skipped when absent
./check.sh flat_pool/moods.pml               # one scenario
GENMC=~/genmc/build/bin/genmc ./check.sh     # both
```

The far model runs only on the litmus shapes, since no pool path posts a relaxed no-return add.
`GENMC_CLANG` names the compiler GenMC was built with, `clang++` by default; `GENMC_SECONDS` caps one client, 300 by default; `GENMC_UNROLL` gives every loop that many turns, 3 by default, since GenMC treats a weak compare-exchange as one that may fail spuriously and a read-first retry loop never ends for it.
Every verdict runs in its own directory, `VERIFY_JOBS` at a time, four by default since each verifier holds a hash table of its own, and the lines print in the order they were queued once the last one lands.

GenMC ships a freestanding C library whose headers shadow the platform's, `atomic` without `std::atomic_ref` among them, and puts its include directory behind the caller's.
The clients need the real standard library, so `genmc_ready` writes a directory of forwarders for the shadowed names and puts it first.
The clients spawn through `__VERIFIER_thread_create`: the platform's `pthread_create` is not intercepted, and `std::thread` rides on it, which is why `flat_pool/client.cpp` spells the pool's words rather than running the pool.
