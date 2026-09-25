# Verification

Model checking for the words ForkUnion's atomics touch, and the memory model every index model in USearch runs under.
Two tools, both open source and neither on the JVM: [Spin](https://spinroot.com) for the protocol layer, [GenMC](https://github.com/MPI-SWS/genmc) for the C++ source under RC11.
`./check.sh` runs everything and compares each verdict with the expected one.

- `weak_memory.pml` — the C++ memory model as views, for Promela: relaxed, acquire, release, `acq_rel`, both fences, release sequences.
- `weak_memory_litmus.pml` — the calibration: the classic shapes, each asserting the outcome RC11 forbids.
  `check.sh` expects exactly RC11's verdicts.
- `flat_pool.pml` — `flat_pool`: the epoch clock, the countdown, what a completed join sees, the moods, a re-spawn, a poll on a stale token, and the C shim's callback slot.
  Scenarios: `-Dscenario=moods`, `respawn`, `polling`, `redispatch`, `stale_join`.
- `flat_pool.cpp` — the same words under GenMC.
- `for_n_dynamic.pml` — `for_n_dynamic`: one private cursor per thread, the coprime steal, the static prongs, the overshoot bound, the join, and no reset in flight.
  Scenarios: `-Dscenario=nested`.
- `distributed_pool.pml` — `distributed_pool`: the lockstep dispatch over colocations, the ANDed completion, the caller-first join, and `spin_mutex` under a slice.
  Scenarios: `-Dscenario=overlap`, `locked`.
- `standard_atomic_ref.cpp` — the portable `fetch_max` loop of `standard_atomic_ref` and the `atomic_fetch_add_if_at_most` loop it forwards to, under GenMC, the real header.
- `genmc.hpp` — what a client takes from GenMC: `spawn`, `join`, `verify`.

## Conventions

Every model opens the same way: the file docblock, the include, then the names.
The comments follow the C++ headers' rules, and the pre-commit hook checks `*.pml` with the same Doxygen checks.
The file docblock opens with `@file`, `@author`, `@date` and a `@brief` of at most three lines, then the body, where each invariant and each scenario is a paragraph opening with its subject.
Every `inline` and `proctype`, and every documented group of `#define`s, carries a `/** */` docblock directly above it whose first sentence is its summary.
A note over a whole `#if` branch is a plain `/* */` above the `#if`, and `//` is kept for notes inside a body.
Inside a block, a single name reads `@c name`, a parameter of the inline below reads `@p name`, and a span of several tokens stays in backticks.
Code is cited by symbol and file, as `flat_pool::unsafe_join` in `flat.hpp`, never by line number, which drifts with every edit above it.
The module fixes its own shape, `thread_count`, `location_count` and `history_depth`, the most writes one word ever receives, the initial one included, as the maxima over every model.
Everything is lowercase snake case; pan's own `-DSAFETY`, `-DCOLLAPSE` and `-DVECTORSZ` in the runner are the only capitals.
A thread index is `<role>_thread`, apart from the process that plays the role; processes that come in numbers name themselves as they start.
A word carries the name of the C++ member it stands for, without the trailing underscore, or a small accessor like `row_lock(id)` when several rows share a shape.
Sizes and values are plain nouns; a parameter a `-D` may override sits under `#ifndef`.
The memory model, a litmus shape and a scenario are each one enumerated knob whose values are integers named in the module or the model and tested with `#if`: `-Dmemory=sequential`, `-Dshape=coherence`, `-Dscenario=admission`; a typo fails the range check instead of reading as zero.
A weakening of the code under test is `without_<what the code has>`, tested with `#ifdef` and always expected to fail.
Field extractors read `<field>_of(word)`, transforms are past participles like `stepped(word)`, per-site orders are `<site>_order`, values made from an id are `<state>(id)`.
USearch's models include `weak_memory.pml` by the same name through a forwarding file beside them.

## Three memory models, one interface

Every model passes its own thread index to `load`, `store`, `read_modify_write`, `read_modify_write_if`, `compare_exchange`, `add_no_return`, `fence_acquire` and `fence_release`, and `-Dmemory=` picks the memory model at `spin -a` time:

- `-Dmemory=sequential`: one copy of every location, every access one step.
The protocol layer, and where logic bugs are found first.
- `-Dmemory=views`, the default: every location is a history of writes, every thread carries a view over the histories, a release write stamps the writer's view onto the write and an acquire read merges it.
This is the view semantics of Kaiser, Dang, Dreyer, Lahav and Vafeiadis without promises, which is RC11's release-acquire-relaxed fragment: load buffering is forbidden, as in RC11.
- `-Dmemory=far`: the views, plus a relaxed no-return add is posted rather than performed.
It lands at some later step in the `far_cache` process, and until then no release by the posting thread carries it; a same-address access by the poster lands it first.
This is RAO-INT as Intel documents it: `AADD` and kin are "implemented using the weakly-ordered memory consistency model of write combining (WC) memory type", and "a fencing operation implemented with LFENCE, SFENCE, or MFENCE instruction should be used in conjunction with AADD if a stronger ordering is required".
Nothing else is promised to order them, and a C++ release fence compiles to no instruction on x86, so a relaxed no-return add followed by a release fence or a release store is posted in the model exactly as it is in silicon.
The ForkUnion reference maps only relaxed no-return operations onto them, and a release order on the same call is a lock-prefixed instruction, which is how the index sites that need the order now spell it.

The calibration in `weak_memory_litmus.pml` pins the model to RC11 on message passing with and without releases and fences, release sequences continued by a read-modify-write and broken by a store, coherence, load buffering, and the far shape: a relaxed no-return add before a release store, carried in C++ and posted under far memory.

## What the models found, and what changed for it

The pool's three atomics are sound, and the models prove the behaviours composed on them; what they found sits at the edges.
A dispatch waking a chilled pool restored the workers' scheduling class with `SCHED_FIFO | SCHED_RR`, which is `SCHED_BATCH` on Linux and a priority the real-time classes refuse anyway, so the workers came back below normal; the nudge is `SCHED_OTHER` now, the class they started on, as the FreeBSD branch beside it already said.
The same wake-up was one `compare_exchange_weak` outside any loop, which may fail spuriously on load-linked architectures; on a flat pool that only costs latency, but a colocated worker still in its startup wait after a `sleep` leaves it only when the mood stops being `chill_k`, and a dispatch whose exchange failed would never release it while its join waited for it forever.
`flat_pool.pml -Dscenario=moods -Dwithout_strong_wake` shows the stuck pair; the exchange is strong now, the same instruction on x86.
`terminate` was documented as callable from any thread at any time, and asserts that no task is running and the last dispatch was joined; the docs say so now.
The C shim's `fu_pool_unsafe_join` cleared its callback slot on any token, so a stale join during a live dispatch left the live join with nothing to run and a worker calling through the cleared slot; `flat_pool.pml -Dscenario=stale_join -Dwithout_generation_check` shows it, and the shim clears the slot only for the generation it published now.

Three things stand as they are, and the models say why.
The release on each cursor in `invoke_for_n_dynamic::reset_slices_` is redundant, since the dispatch's release on the epoch carries every cursor and end to every worker; `-Dpublish_relaxed` passes to show it, and the release stays.
The workers re-check the mood only under a capped wait, which bounds a `sleep` or a `terminate` notice to one timeout; `-Dwithout_wait_cap` is the uncapped monitor, and a stuck worker.
The epoch's width aliases at the debug widths, as the header prices: `-Depoch_modulus=2` under `-Dscenario=polling` makes a stale token name the live generation, and the stale join contributes a slice that is not its own.
A `broadcast_join` kept alive across a `terminate` and a `spawn` would alias after one re-spawn rather than after 2^bits epochs, since `terminate` resets the epoch; it asserts every dispatch joined, so nothing outlives it by contract.

## Running

```sh
./check.sh                                   # Spin only, GenMC skipped when absent
GENMC=~/genmc/build/bin/genmc ./check.sh     # both
```

The suite's 66 verdicts take about twenty seconds four at a time, which is the default; the far model is skipped throughout, since no pool path posts a relaxed no-return add.
`GENMC_CLANG` names the compiler GenMC was built with, `clang++` by default; `GENMC_SECONDS` caps one client, 300 by default; `GENMC_UNROLL` gives every loop that many turns, 3 by default, since GenMC treats a weak compare-exchange as one that may fail spuriously and a read-first retry loop never ends for it.
Every verdict runs in its own directory, `VERIFY_JOBS` at a time, four by default since each verifier holds a hash table of its own, and the lines print in the order they were queued once the last one lands.

GenMC ships a freestanding C library whose headers shadow the platform's, `atomic` without `std::atomic_ref` among them, and puts its include directory behind the caller's.
The clients need the real standard library, so `genmc_ready` writes a directory of forwarders for the shadowed names and puts it first.
The clients spawn through `__VERIFIER_thread_create`: the platform's `pthread_create` is not intercepted, and `std::thread` rides on it, which is why `flat_pool.cpp` spells the pool's words rather than running the pool.

A verdict is `pass` when no assertion fails, `fail` when the model admits the outcome the assertion forbids, and `check.sh` expects `fail` exactly where RC11 allows the outcome or a variant drops the fence the protocol needs.
`-Dwithout_decrement_acquire` on `flat_pool` is such a variant: with the countdown's decrements weakened from `acq_rel` to release, the last contributor no longer acquires its peers, and the join reads a stale result.
