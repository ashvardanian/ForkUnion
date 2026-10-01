#!/usr/bin/env bash
# Runs every Promela model under every memory model it is meant for, and every GenMC client, and
# compares each verdict with the expected one: `pass` means no assertion fails, `fail` means the
# model admits the outcome the assertion forbids - the litmus shapes RC11 allows, the variants
# that drop a fence or shrink a modulus. USearch's `check.sh` sources this one for its functions.
# Needs `spin` and a C compiler; a GenMC on the path, or named by `GENMC`, also runs the clients.

set -u
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) # `genmc.hpp` lives beside this file, wherever it is run from
# Every verdict runs in its own directory in the background, `VERIFY_JOBS` at a time - four by
# default, since each pan holds a hash table of its own; `finish` prints them in the order they
# were queued, so the output reads like a serial run.
jobs_limit=${VERIFY_JOBS:-4}
queued=0

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$jobs_limit" ]; do sleep 0.1; done; }

# section <title>: a heading, printed in its turn
section() {
    mkdir -p "$build/$queued"
    echo "$1" >"$build/$queued/result"
    queued=$((queued + 1))
}

# verify <model.pml> <expected> <spin defines...>
verify() {
    local model=$1 expected=$2 dir=$build/$queued
    shift 2
    mkdir -p "$dir"
    queued=$((queued + 1))
    throttle
    (
        local verdict
        if ! (cd "$dir" && spin -a "$@" "$OLDPWD/$model" >pan.out 2>&1 &&
            cc -O2 -DSAFETY -DCOLLAPSE -DVECTORSZ=4096 -w -o pan pan.c >>pan.out 2>&1); then verdict=broken
        else
            (cd "$dir" && ./pan -m1000000 -w26 >>pan.out 2>&1)
            if grep -q "errors: 0" "$dir/pan.out"; then verdict=pass
            elif grep -qE "errors: [1-9]" "$dir/pan.out"; then verdict=fail
            else verdict=broken; fi
        fi
        report "$verdict" "$expected" "$model $*" "$dir/pan.out" >"$dir/result"
    ) &
}

report() { # report <verdict> <expected> <label> <log>
    if [ "$1" = "$2" ]; then echo "  ok    $3"
    else echo "  WRONG $3 : expected $2, got $1"; sed -n 1,12p "$4"; touch "$(dirname "$4")/wrong"; fi
}

# finish: waits for every verdict, prints them in order, and exits with the count of wrong ones
finish() {
    wait
    local index failed=0
    for ((index = 0; index < queued; index++)); do
        cat "$build/$index/result"
        [ -e "$build/$index/wrong" ] && failed=$((failed + 1))
    done
    exit $failed
}

# GenMC ships a freestanding C library whose headers shadow the platform's - `atomic` without
# `std::atomic_ref`, `stdio.h` without `EOF` - and puts its include directory behind the caller's.
# The clients need the real standard library, so a directory of forwarders for the shadowed
# names goes first. The compiler asked is the one GenMC was built with, `clang++` by default.
genmc_ready() {
    genmc=${GENMC:-$(command -v genmc || true)}
    [ -n "$genmc" ] || { section "  skip  GenMC is not installed; set GENMC to its executable"; return 1; }
    local compiler=${GENMC_CLANG:-clang++} directories libcxx="" libc=""
    directories=$(echo | "$compiler" -x c++ -E -v - 2>&1 | sed -n '/<...> search starts here/,/End of search list/p' | grep '^ /' | awk '{print $1}')
    for directory in $directories; do
        [ -z "$libcxx" ] && [ -f "$directory/atomic" ] && libcxx=$directory && continue
        [ -z "$libc" ] && [ -f "$directory/stdio.h" ] && libc=$directory
    done
    shadow=$build/shadow
    mkdir -p "$shadow"
    local name guard
    for name in atomic thread cassert cstdio cstdlib; do printf '#include "%s/%s"\n' "$libcxx" "$name" >"$shadow/$name"; done
    for name in assert.h pthread.h; do printf '#include "%s/%s"\n' "$libc" "$name" >"$shadow/$name"; done
    for name in stdio.h stdlib.h errno.h; do
        guard=_LIBCPP_$(echo "$name" | tr 'a-z.' 'A-Z_')
        printf '#define %s\n#include "%s/%s"\n' "$guard" "$libc" "$name" >"$shadow/$name"
    done
}

# GenMC treats a weak compare-exchange as one that may fail spuriously, so a read-first retry
# loop never ends for it; `--unroll` gives every loop that many turns.
genmc_unroll=${GENMC_UNROLL:-3}

# Runs a command with a wall-clock limit, since an exhaustive exploration that grows past a few
# minutes is a client to shrink, not a result to wait for.
limited() { # limited <seconds> <command...>
    local seconds=$1
    shift
    "$@" &
    local pid=$!
    (sleep "$seconds" && kill "$pid" 2>/dev/null) &
    local watchdog=$!
    wait "$pid" 2>/dev/null
    local status=$?
    pkill -P "$watchdog" 2>/dev/null
    kill "$watchdog" 2>/dev/null
    return $status
}

# verify_client <client.cpp> <expected> <compiler flags...>
verify_client() {
    local client=$1 expected=$2 dir=$build/$queued unroll=$genmc_unroll
    shift 2
    mkdir -p "$dir"
    queued=$((queued + 1))
    throttle
    (
        local verdict
        limited "${GENMC_SECONDS:-300}" "$genmc" --disable-estimation --unroll="$unroll" -- -std=c++23 -I"$shadow" -I"$harness" "$@" "$client" >"$dir/genmc.out" 2>&1 ||
            echo "GenMC stopped: exit status $? after ${GENMC_SECONDS:-300}s or an error" >>"$dir/genmc.out"
        if grep -q "No errors were detected" "$dir/genmc.out"; then verdict=pass
        elif grep -qE "^Error|Assertion violation|: ERROR " "$dir/genmc.out"; then verdict=fail
        else verdict=broken; fi
        report "$verdict" "$expected" "$client $*" "$dir/genmc.out" >"$dir/result"
    ) &
}

# Sourced by USearch's `check.sh` for the functions above; run directly, it checks ForkUnion.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
cd "$(dirname "$0")"

section "weak_memory_litmus.pml: the memory model against the shapes RC11 decides"
for shape in message_passing_without_release:fail message_passing_release_acquire:pass message_passing_fences:pass \
    release_sequence_read_modify_write:pass release_sequence_store:fail coherence:pass load_buffering:pass; do
    verify weak_memory_litmus.pml pass -Dmemory=sequential -Dshape="${shape%%:*}"
    verify weak_memory_litmus.pml "${shape##*:}" -Dshape="${shape%%:*}"
    verify weak_memory_litmus.pml "${shape##*:}" -Dmemory=far -Dshape="${shape%%:*}"
done
verify weak_memory_litmus.pml pass -Dmemory=sequential -Dshape=far_add_before_release
verify weak_memory_litmus.pml pass -Dshape=far_add_before_release
verify weak_memory_litmus.pml fail -Dmemory=far -Dshape=far_add_before_release

section "for_n_dynamic.pml: private cursors, the coprime steal, the static prongs, the overshoot bound, the join, and no reset in flight"
verify for_n_dynamic.pml pass -Dmemory=sequential
verify for_n_dynamic.pml pass
verify for_n_dynamic.pml pass -Dtasks=2
verify for_n_dynamic.pml pass -Dpublish_relaxed
verify for_n_dynamic.pml fail -Dwithout_dispatch_release
verify for_n_dynamic.pml fail -Dwithout_completion_acquire
verify for_n_dynamic.pml fail -Dmemory=sequential -Dwithout_claim_guard
verify for_n_dynamic.pml pass -Dmemory=sequential -Dwithout_probe
verify for_n_dynamic.pml pass -Dmemory=sequential -Dwithout_single_visit
verify for_n_dynamic.pml fail -Dmemory=sequential -Dwithout_probe -Dwithout_single_visit
verify for_n_dynamic.pml fail -Dmemory=sequential -Dscenario=nested

section "distributed_pool.pml: the lockstep dispatch, the ANDed completion, the caller-first join, and spin_mutex"
verify distributed_pool.pml pass -Dmemory=sequential -Dgenerations=2
verify distributed_pool.pml pass
verify distributed_pool.pml fail -Dmemory=sequential -Dwithout_dispatch_before_join
verify distributed_pool.pml pass -Dmemory=sequential -Dgenerations=2 -Dscenario=overlap
verify distributed_pool.pml fail -Dmemory=sequential -Dscenario=overlap -Dwithout_caller_first_join
verify distributed_pool.pml pass -Dscenario=locked
verify distributed_pool.pml fail -Dscenario=locked -Dwithout_lock_acquire
verify distributed_pool.pml fail -Dscenario=locked -Dwithout_unlock_release

section "flat_pool.pml: the epoch clock, the countdown, what a completed join sees, and the C shim's callback slot"
verify flat_pool.pml pass -Dmemory=sequential
verify flat_pool.pml pass
verify flat_pool.pml fail -Dwithout_decrement_acquire
verify flat_pool.pml pass -Dmemory=sequential -Dscenario=moods
verify flat_pool.pml pass -Dscenario=moods
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=moods -Dwithout_wait_cap
verify flat_pool.pml fail -Dscenario=moods -Dwithout_chill_at_spawn
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=moods -Dwithout_strong_wake
verify flat_pool.pml pass -Dscenario=respawn
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=respawn -Dwithout_join_before_reset
verify flat_pool.pml pass -Dmemory=sequential -Dscenario=polling
verify flat_pool.pml pass -Dscenario=polling -Dgenerations=1
verify flat_pool.pml fail -Dscenario=polling -Dgenerations=1 -Dwithout_complete_acquire
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=polling -Depoch_modulus=2
verify flat_pool.pml pass -Dmemory=sequential -Dscenario=redispatch
verify flat_pool.pml pass -Dscenario=redispatch
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=redispatch -Dwithout_join_before_dispatch
verify flat_pool.pml pass -Dmemory=sequential -Dscenario=stale_join
verify flat_pool.pml pass -Dscenario=stale_join
verify flat_pool.pml fail -Dmemory=sequential -Dscenario=stale_join -Dwithout_generation_check

section "GenMC clients: the portable conditional and extremal loops, and the fork-join words"
if genmc_ready; then
    verify_client standard_atomic_ref.cpp pass -I../include
    verify_client flat_pool.cpp pass
    verify_client flat_pool.cpp fail -Dwithout_decrement_acquire
fi

finish
