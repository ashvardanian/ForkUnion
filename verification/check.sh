#!/usr/bin/env bash
# Runs every `@verify` line of every Promela scenario and GenMC client and compares each verdict
# with the expected one: `pass` means the search completed with no error, `fail` that an assertion
# was violated, `stuck` that a process waits forever. Needs `spin` and a C compiler; a GenMC on the
# path, or named by `GENMC`, also runs the clients.

set -u
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) # `genmc.hpp` lives beside this file, wherever it is run from
# Every verdict runs in its own directory in the background. A model's `@budget` line names the
# megabytes its search may take, `default_budget_megabytes` without one: pan stops itself there, so
# a model grown too large reads as broken, and its hash table takes half of it. Verdicts start while
# their budgets fit in `VERIFY_MEMORY_MEGABYTES` together, three quarters of the available memory
# by default, and at most `VERIFY_JOBS` at once; a model whose budget alone exceeds that pool is
# skipped. `finish` prints verdicts in the order they were queued.
default_budget_megabytes=1024
if [ -r /proc/meminfo ]; then available_megabytes=$(awk '/^MemAvailable:/ {print int($2 / 1024)}' /proc/meminfo)
else available_megabytes=$(($(sysctl -n hw.memsize) / 1048576)); fi
pool_megabytes=${VERIFY_MEMORY_MEGABYTES:-$((available_megabytes * 3 / 4))}
jobs_limit=${VERIFY_JOBS:-$(getconf _NPROCESSORS_ONLN)}
queued=0
running_pids=()
running_budgets=()

# model_budget <file>: the megabytes its `@budget` line grants, or the default
model_budget() {
    local budget
    budget=$(sed -n 's/^ \*  @budget \([0-9]*\).*/\1/p' "$1" | head -1)
    echo "${budget:-$default_budget_megabytes}"
}

# hash_bits <megabytes>: the `-w` that gives pan's hash table half the budget at eight bytes a slot,
# but never past a gigabyte, so a large budget pays only for the states it stores
hash_bits() {
    local megabytes=$1 bits=16
    while [ "$megabytes" -gt 1 ] && [ "$bits" -lt 27 ]; do megabytes=$((megabytes / 2)) bits=$((bits + 1)); done
    echo "$bits"
}

# throttle <megabytes>: waits until a verdict of that budget fits beside the running ones; one
# verdict alone always starts, even when its budget exceeds the pool
throttle() {
    local index used count
    while :; do
        used=0 count=0
        for index in "${!running_pids[@]}"; do
            if kill -0 "${running_pids[$index]}" 2>/dev/null; then
                used=$((used + running_budgets[index])) count=$((count + 1))
            else unset "running_pids[$index]" "running_budgets[$index]"; fi
        done
        [ "$count" -eq 0 ] && return
        [ "$count" -lt "$jobs_limit" ] && [ $((used + $1)) -le "$pool_megabytes" ] && return
        sleep 0.1
    done
}

# admit <megabytes>: records the verdict just started in the background against the pool
admit() { running_pids+=("$!") running_budgets+=("$1"); }

# section <title>: a heading, printed in its turn
section() {
    mkdir -p "$build/$queued"
    echo "$1" >"$build/$queued/result"
    queued=$((queued + 1))
}

# verify <model.pml> <expected> <spin defines...>
verify() {
    local model=$1 expected=$2 dir=$build/$queued budget
    shift 2
    mkdir -p "$dir"
    queued=$((queued + 1))
    budget=$(model_budget "$model")
    if [ "$budget" -gt "$pool_megabytes" ]; then
        echo "  skip  $model $* : needs $budget MB, $pool_megabytes MB available" >"$dir/result"
        return
    fi
    throttle "$budget"
    (
        local verdict
        # The harness's directory on the include path lets a repo sourcing this runner keep only its models.
        # Every include is an angle-bracket path from the harness or from this repo's verification root.
        if ! (cd "$dir" && spin -a -E-I"$harness" -E-I"$OLDPWD" "$@" "$OLDPWD/$model" >pan.out 2>&1 &&
            cc -O2 -DSAFETY -DCOLLAPSE -DVECTORSZ=4096 -DMEMLIM="$budget" -w -o pan pan.c >>pan.out 2>&1); then
            verdict=broken
        else
            (cd "$dir" && ./pan -m1000000 -w"$(hash_bits "$budget")" >>pan.out 2>&1)
            # A cut-short search, a knob defined twice, a write or index past a scenario's shape, or a
            # search out of memory proves nothing
            if grep -qE "macro redefined|max search depth too small|invalid array index|violated.*history_full|MEMLIM|out of memory" "$dir/pan.out"; then verdict=broken
            elif grep -q "errors: 0" "$dir/pan.out"; then verdict=pass
            elif grep -q "pan:1: assertion violated" "$dir/pan.out"; then verdict=fail
            elif grep -q "pan:1: invalid end state" "$dir/pan.out"; then verdict=stuck
            else verdict=broken; fi
        fi
        [ "$verdict" = pass ] && [ -n "$cyclic" ] && verdict=$(cycles "$model" "$dir/cycles" "$@")
        report "$verdict" "$expected" "$model $*" "$dir/pan.out" >"$dir/result"
    ) &
    admit "$budget"
}

# cycles <model.pml> <dir> <spin defines...>: `livelock` when some run of the scenario never ends under weak
# fairness and eventual visibility, else `pass`. Every scenario's threads finish, so any cycle is one, and none
# needs a progress label.
cycles() {
    local model=$1 dir=$2
    shift 2
    mkdir -p "$dir"
    [[ " $* " == *" -Dstaleness_bound="* ]] || set -- "$@" -Dstaleness_bound=2
    if ! (cd "$dir" && spin -a -E-I"$harness" -E-I"$OLDPWD" "$@" "$OLDPWD/$model" >pan.out 2>&1 &&
        cc -O2 -DNP -DCOLLAPSE -DVECTORSZ=4096 -DMEMLIM="$budget" -w -o pan pan.c >>pan.out 2>&1); then echo broken
    else
        (cd "$dir" && ./pan -l -f -m1000000 -w"$(hash_bits "$budget")" >>pan.out 2>&1)
        if grep -qE "macro redefined|max search depth too small|invalid array index|violated|MEMLIM|out of memory" "$dir/pan.out"; then echo broken
        elif grep -q "pan:1: non-progress cycle" "$dir/pan.out"; then echo livelock
        elif grep -q "errors: 0" "$dir/pan.out"; then echo pass
        else echo broken; fi
    fi
}

report() { # report <verdict> <expected> <label> <log>
    # Spin's own tally of the memory a search took, so a model growing toward the budget shows early
    local megabytes
    megabytes=$(grep -oE '^[[:space:]]*[0-9.]+[[:space:]]+total actual memory usage' "$4" | awk '{print $1}')
    if [ "$1" = "$2" ]; then echo "  ok    $3${megabytes:+ (${megabytes} MB)}"
    else echo "  WRONG $3 : expected $2, got $1"; sed -n 1,12p "$4"; touch "$(dirname "$4")/wrong"; fi
}

# finish: waits for every verdict, prints them in order, and exits with the count of wrong ones
finish() {
    wait
    local index failed=0
    for ((index = 0; index < queued; index++)); do
        cat "$build/$index/result" >>"$build/results"
        [ -e "$build/$index/wrong" ] && failed=$((failed + 1))
    done
    cat "$build/results"
    [ -z "${GITHUB_STEP_SUMMARY:-}" ] || printf '```\n%s\n```\n' "$(cat "$build/results")" >>"$GITHUB_STEP_SUMMARY"
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

# verify_client <client.cpp> <expected> <memory model> <entry>: GenMC exits 0 when it found no
# error and 42 when it found one; `--check-liveness` makes a wait that never ends one of them.
verify_client() {
    local client=$1 expected=$2 memory=$3 entry=$4 dir=$build/$queued unroll=$genmc_unroll budget
    mkdir -p "$dir"
    queued=$((queued + 1))
    budget=$(model_budget "$client")
    throttle "$budget"
    (
        local verdict
        limited "${GENMC_SECONDS:-300}" "$genmc" --"$memory" --check-liveness --disable-estimation --unroll="$unroll" \
            --program-entry-function="$entry" -- -std=c++23 -I"$shadow" -I"$harness" -I"$PWD" -I"$PWD/../include" -I"$PWD/../c" -I"$harness/../include" -I"$harness/../c" \
            "$client" >"$dir/genmc.out" 2>&1
        case $? in
        0) if grep -q "complete executions explored: 0" "$dir/genmc.out"; then verdict=broken; else verdict=pass; fi ;;
        42) if grep -q "Liveness violation" "$dir/genmc.out"; then verdict=stuck; else verdict=fail; fi ;;
        *) verdict=broken ;;
        esac
        report "$verdict" "$expected" "$client $memory $entry" "$dir/genmc.out" >"$dir/result"
    ) &
    admit "$budget"
}

# citations_unresolved <file...>: every cited path, name or `name` in `file` pair the code no longer has, each as
# ` file:citation`. A path resolves against the repository or the harness; a snake_case name against a word of the
# code, or of the code in a model, client or script, its comments dropped; a pair against a word of the named file.
citations_unresolved() {
    local tracked words file cited name path sources
    tracked=$(git -C .. ls-files 2>/dev/null) || return 0
    words=$( (git -C .. grep -ohwI -E '[A-Za-z_][A-Za-z0-9_]*' -- ':!verification'
        cat ./*.pml ./*/*.pml ./*.cpp ./*/*.cpp "$harness"/*.pml "$harness"/*.hpp "$harness"/*.sh 2>/dev/null |
            sed -E 's|//.*||; s|#[[:space:]].*||; /^[[:space:]]*(\/\*|\*)/d' | grep -ohwE '[A-Za-z_][A-Za-z0-9_]*') | sort -u)
    for file in "$@"; do
        [ -f "$file" ] || continue
        for cited in $(grep -oE '(`|@c )[A-Za-z0-9_./-]+\.(hpp|h|c|cpp|cu|cuh|hip|metal|pml|sh)\b`?' "$file" | sed 's/^@c //' | tr -d '`' | sort -u); do
            grep -qE "(^|/)${cited//./\\.}\$" <<<"$tracked" || [ -e "$harness/$cited" ] || echo -n " $file:$cited"
        done
        for cited in $( (grep -oE '`[a-z][a-z0-9]*_[a-z0-9_]*`' "$file" | tr -d '`'
            grep -oE '@c [a-z][a-z0-9_]*_\b' "$file" | cut -c4-) | sort -u); do
            grep -qx "$cited" <<<"$words" || echo -n " $file:$cited"
        done
        while read -r name path; do
            sources=$(grep -E "(^|/)${path//./\\.}\$" <<<"$tracked" | sed 's|^|../|')
            [ -n "$sources" ] && grep -qw -- "${name##*::}" $sources || echo -n " $file:$name@$path"
        done < <(grep -oE '(`|@c )[A-Za-z_~:0-9]+`? (in|from) (`|@c )[A-Za-z0-9_./-]+\.[a-z]+' "$file" |
            sed -E 's/@c //g; s/`//g; s/ (in|from) / /')
    done
}

# verify_all [file...]: every `@verify` line of the given scenarios and clients, or of every one
# here and one directory down, then the verdicts in order. A line names the expected verdict and
# the memory models, then knob overrides for a scenario or the entry for a client. Every knob the
# files declare must be set on some line, and every knob a line sets must be declared.
verify_all() {
    local files=("$@") file expected memories rest memory declared used unmatched entries field entry
    # Both roots share one include path, so a repo file named like a harness file would shadow it
    if [ "$PWD" != "$harness" ]; then
        for file in weak_memory.pml monitor_wait.pml genmc.hpp; do
            [ ! -e "$file" ] || { echo "verification/$file shadows ForkUnion's harness file of that name; rename it" >&2; exit 1; }
        done
    fi
    genmc_ready
    [ ${#files[@]} -gt 0 ] || files=(*.pml */*.pml *.cpp */*.cpp)
    declared=$(sed -n 's/^#ifndef \([a-z0-9_]*\)$/\1/p' "${files[@]}" */protocol.pml *.pml 2>/dev/null | grep -vx memory | sort -u)
    # Knobs of the harness itself, which a sourcing repo may set without declaring and need not set
    offered=$(sed -n 's/^#ifndef \([a-z0-9_]*\)$/\1/p' "$harness/monitor_wait.pml" "$harness/weak_memory.pml" | grep -vx memory | sort -u)
    used=$(sed -n 's/^ \*  @verify [a-z]* [a-z0-9,]* \([^:]*\).*/\1/p' "${files[@]}" 2>/dev/null | tr ' ' '\n' | sed -n 's/=.*//p' | sort -u)
    [ $# -gt 0 ] && declared=$(comm -12 <(echo "$declared") <(echo "$used"))
    unmatched=$( (comm -23 <(echo "$declared") <(echo "$used"); comm -13 <(sort -u <(echo "$declared"; echo "$offered")) <(echo "$used")) | xargs)
    # A client's knobs are its `knobs_t` fields, each named by some entry, and every entry exists
    for file in "${files[@]}"; do
        [[ $file == *.cpp ]] && grep -qs '^ \*  @verify' "$file" || continue
        entries=$(sed -n 's/^ \*  @verify [a-z]* [a-z0-9,]* \([a-z0-9_]*\).*/\1/p' "$file" | sort -u)
        for field in $(sed -n '/^struct knobs_t {/,/^};/s/^ *[a-z_:]* \([a-z0-9_]*\) = .*;$/\1/p' "$file"); do
            grep -q "_${field}_" <<<"$entries" || unmatched="$unmatched $file:$field"
        done
        for entry in $entries; do grep -q "^extern \"C\" int $entry()" "$file" || unmatched="$unmatched $file:$entry"; done
    done
    unmatched="$unmatched$(citations_unresolved "${files[@]}")"
    mkdir -p "$build/$queued"
    if [ -z "$unmatched" ]; then echo "  ok    every knob is declared and set, every entry exists, every citation resolves" >"$build/$queued/result"
    else
        echo "  WRONG knobs, entries or citations without a counterpart: $unmatched" >"$build/$queued/result"
        touch "$build/$queued/wrong"
    fi
    queued=$((queued + 1))
    for file in "${files[@]}"; do
        grep -qs '^ \*  @verify' "$file" || continue
        [[ $file == *.cpp && -z "${genmc:-}" ]] && continue
        section "$file: $(sed -n 's/^ \*  @brief //p' "$file" | head -1)"
        # A scenario that expects a livelock on some line searches every line for cycles too
        cyclic=$(grep -s '^ \*  @verify livelock' "$file" || true)
        while read -r expected memories rest; do
            for memory in ${memories//,/ }; do
                case $file in
                *.pml) verify "$file" "$expected" -Dmemory="$memory" $(for knob in $rest; do echo "-D$knob"; done) ;;
                *.cpp) verify_client "$file" "$expected" "$memory" "$rest" ;;
                esac
            done
        done < <(sed -n 's/^ \*  @verify \([^:]*\).*/\1/p' "$file")
    done
    finish
}

# Sourced, it only defines the functions above; run directly, it checks the directory it lives in.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
cd "$(dirname "$0")"
verify_all "$@"
