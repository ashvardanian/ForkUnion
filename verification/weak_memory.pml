/**
 *  @file verification/weak_memory.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief The C++ memory model as views, for Promela.
 *
 *  Every atomic location is a history of writes, and every thread carries three views over the
 *  histories: @c current, the oldest write it may still read per location; @c released, the view
 *  its last release fence captured; and @c acquired, what its relaxed loads have picked up and its
 *  next acquire fence will merge. A release write stamps the writer's view onto the write, an
 *  acquire read merges that stamp into the reader. Read-modify-writes read the newest write only
 *  and carry its stamp forward, which is the release sequence. This is the view semantics of
 *  Kaiser, Dang, Dreyer, Lahav and Vafeiadis, without promises, which makes it exactly RC11's
 *  release-acquire-relaxed fragment: load buffering is forbidden, as in RC11.
 *
 *  Three memory models share one interface, chosen by `-Dmemory=` at `spin -a` time:
 *
 *  - @c sequential: one copy of every location, every access a single step.
 *  - @c views, the default: the views above.
 *  - @c far: the views, plus a relaxed no-return add is posted rather than performed. The bit
 *    forms RAO-INT posts the same way, @c aand, @c aor and @c axor, have no inline here, since no
 *    model needs one; a model that adds one follows @c add_no_return.
 *
 *  The @c sequential model is the protocol layer: fast, and the place to find logic bugs first.
 *
 *  The @c far model lands a posted add in the history at some later step, in the @c far_cache
 *  process, and until then no release by the posting thread carries it. Same-address accesses by
 *  the poster land it first. This is RAO-INT as Intel documents it: @c aadd runs under the
 *  write-combining ordering, and only SFENCE or MFENCE order it, which a C++ release fence never
 *  emits on x86.
 *
 *  The module fixes its own shape - @c thread_count, @c location_count, and @c history_depth, the
 *  most writes one location may ever receive, the initial one included - as the maxima over every
 *  model; a model only names its threads and words and passes its thread index to every access.
 *  Orders are the four the indexes use; nothing here is sequentially consistent, since no index
 *  site asks for it.
 */

#define thread_count 5
#define location_count 9
#define history_depth 16

/** The knob's values are integers, so a typo in `-Dmemory=` fails the range check below instead of
 *  reading as zero inside `#if`. */
#define sequential 1
#define views 2
#define far 3
#ifndef memory
#define memory views
#endif
#if memory < sequential || memory > far
#error "memory is sequential, views or far"
#endif

/** The four orders the indexes use, as bits: the low one acquires, the high one releases. */
#define order_relaxed 0
#define order_acquire 1
#define order_release 2
#define order_acq_rel 3
#define acquires(order) ((order) & 1)
#define releases(order) ((order) & 2)

/*  Under @c sequential every location is one word and every access one atomic step, so each inline
 *  below is the interface the views further down implement, with the orders dropped. */
#if memory == sequential

int words[location_count];

/** Reads location @p l into @p out. */
inline load(t, l, order, out) { atomic { out = words[l] } }

/** Writes @p value to location @p l. */
inline store(t, l, order, value) { atomic { words[l] = value } }

/** Reads location @p l into @p observed and writes @p updated in the same step. */
inline read_modify_write(t, l, order, observed, updated) {
    atomic { observed = words[l]; words[l] = updated }
}

/** Reads location @p l into @p observed and writes @p updated only under @p condition. */
inline read_modify_write_if(t, l, order, condition, observed, updated) {
    atomic { observed = words[l]; if :: condition -> words[l] = updated :: else fi }
}

/** Swaps @p desired in where @p l holds @p expected, else reads it into @p expected. */
inline compare_exchange(t, l, order, expected, desired, succeeded) {
    atomic {
        if
        :: words[l] == expected -> words[l] = desired; succeeded = true
        :: else -> expected = words[l]; succeeded = false
        fi
    }
}

/** Adds @p operand to location @p l, reading nothing back. */
inline add_no_return(t, l, order, operand) { atomic { words[l] = words[l] + operand } }

/** Nothing to merge, with one copy of every location. */
inline fence_acquire(t) { skip }

/** Nothing to capture, with one copy of every location. */
inline fence_release(t) { skip }

/** The value location @p l holds now, for a guard or an assertion that adds no access. */
#define newest_value(l) words[l]

/** Nothing is ever posted, so nothing waits to land. */
inline landed(t) { skip }

#else

int history_value[location_count * history_depth];
byte history_view[location_count * history_depth * location_count];
byte newest[location_count];
byte current[thread_count * location_count];
byte released[thread_count * location_count];
byte acquired[thread_count * location_count];
byte view_location; // scratch for the view loops: every access is one atomic step
byte picked;        // scratch: the write a load picked
int summed;         // scratch: the value a no-return add read

#define write_index(l, i) ((l) * history_depth + (i))
#define view_index(l, i, m) (((l) * history_depth + (i)) * location_count + (m))
#define thread_index(t, m) ((t) * location_count + (m))
#define newest_value(l) history_value[write_index(l, newest[l])]

/** The thread's view joins the stamp of write @p i on location @p l into @c current,
 *  for an acquire. */
inline merge_current(t, l, i) {
    for (view_location : 0 .. location_count - 1) {
        if
        :: history_view[view_index(l, i, view_location)] > current[thread_index(t, view_location)] ->
            current[thread_index(t, view_location)] = history_view[view_index(l, i, view_location)]
        :: else
        fi
    };
    view_location = 0
}

/** The thread's view joins the stamp of write @p i on location @p l into @c acquired, for
 *  a relaxed read. */
inline merge_acquired(t, l, i) {
    for (view_location : 0 .. location_count - 1) {
        if
        :: history_view[view_index(l, i, view_location)] > acquired[thread_index(t, view_location)] ->
            acquired[thread_index(t, view_location)] = history_view[view_index(l, i, view_location)]
        :: else
        fi
    };
    view_location = 0
}

/** The thread's view joins the stamp of write @p i on location @p l: into @c current for an
 *  acquire, else into @c acquired. */
inline merge(t, l, i, order) {
    if
    :: acquires(order) -> merge_current(t, l, i)
    :: else -> merge_acquired(t, l, i)
    fi
}

/** The stamp of a fresh write @p i on @p l by @p t: the whole view if it releases, else
 *  the fence view. */
inline stamp(t, l, i, order) {
    for (view_location : 0 .. location_count - 1) {
        if
        :: releases(order) -> history_view[view_index(l, i, view_location)] = current[thread_index(t, view_location)]
        :: else -> history_view[view_index(l, i, view_location)] = released[thread_index(t, view_location)]
        fi
    };
    history_view[view_index(l, i, l)] = i;
    view_location = 0
}

/** Write @p i on @p l carries everything write `i - 1` carried: read-modify-writes
 *  extend release sequences. */
inline inherit(l, i) {
    for (view_location : 0 .. location_count - 1) {
        if
        :: history_view[view_index(l, i - 1, view_location)] > history_view[view_index(l, i, view_location)] ->
            history_view[view_index(l, i, view_location)] = history_view[view_index(l, i - 1, view_location)]
        :: else
        fi
    };
    view_location = 0
}

/** Appends @p value to the history of @p l as its newest write, stamped by @p t. */
inline append(t, l, order, value) {
    assert(newest[l] + 1 < history_depth);
    newest[l] = newest[l] + 1;
    history_value[write_index(l, newest[l])] = value;
    current[thread_index(t, l)] = newest[l];
    stamp(t, l, newest[l], order)
}

#if memory == far
byte pending_location[thread_count]; // one past the location of the posted add; zero when none
int pending_operand[thread_count];

/** Lands the add thread @p t posted: the newest write plus the operand, carrying the
 *  release sequence on. */
inline land(t) {
    atomic {
        assert(pending_location[t] != 0);
        assert(newest[pending_location[t] - 1] + 1 < history_depth);
        newest[pending_location[t] - 1] = newest[pending_location[t] - 1] + 1;
        history_value[write_index(pending_location[t] - 1, newest[pending_location[t] - 1])] =
            history_value[write_index(pending_location[t] - 1, newest[pending_location[t] - 1] - 1)] + pending_operand[t];
        for (view_location : 0 .. location_count - 1) {
            history_view[view_index(pending_location[t] - 1, newest[pending_location[t] - 1], view_location)] = 0
        };
        view_location = 0;
        history_view[view_index(pending_location[t] - 1, newest[pending_location[t] - 1], pending_location[t] - 1)] =
            newest[pending_location[t] - 1];
        inherit(pending_location[t] - 1, newest[pending_location[t] - 1]);
        // The poster did perform it, so its own later same-address reads see it
        current[thread_index(t, pending_location[t] - 1)] = newest[pending_location[t] - 1];
        pending_location[t] = 0;
        pending_operand[t] = 0
    }
}

/** Lands the add thread @p t posted when it went to location @p l, before @p t touches
 *  @p l again. */
inline land_if_same_word(t, l) {
    if
    :: pending_location[t] == (l) + 1 -> land(t)
    :: else
    fi
}

/** Blocks until the thread's posted add landed: the last line of every far-memory scenario. */
inline landed(t) { pending_location[t] == 0 }

/** The far cache: lands any thread's posted add at any step. */
active proctype far_cache() {
end:
    do
    :: atomic { pending_location[0] != 0 -> land(0) }
    :: atomic { pending_location[1] != 0 -> land(1) }
    :: atomic { pending_location[2] != 0 -> land(2) }
    :: atomic { pending_location[3] != 0 -> land(3) }
    :: atomic { pending_location[4] != 0 -> land(4) }
    od
}
#else
inline land_if_same_word(t, l) { skip }
inline landed(t) { skip }
#endif

/** Reads any write from the thread's view onward: this is where staleness lives. */
inline load(t, l, order, out) {
    atomic {
        land_if_same_word(t, l);
        picked = current[thread_index(t, l)];
        do
        :: picked < newest[l] -> picked = picked + 1
        :: break
        od;
        out = history_value[write_index(l, picked)];
        current[thread_index(t, l)] = picked;
        merge(t, l, picked, order);
        picked = 0
    }
}

/** Appends @p value to the history of @p l, stamped as @p order says. */
inline store(t, l, order, value) {
    atomic {
        land_if_same_word(t, l);
        append(t, l, order, value)
    }
}

/** Reads the newest write, so nothing slips between the read and the write. */
inline read_modify_write(t, l, order, observed, updated) {
    atomic {
        land_if_same_word(t, l);
        observed = newest_value(l);
        merge(t, l, newest[l], order);
        append(t, l, order, updated);
        inherit(l, newest[l])
    }
}

/** The @c fetch_add_if_at_most kin: the write happens only under @p condition, spelled
 *  over @p observed. */
inline read_modify_write_if(t, l, order, condition, observed, updated) {
    atomic {
        land_if_same_word(t, l);
        observed = newest_value(l);
        merge(t, l, newest[l], order);
        current[thread_index(t, l)] = newest[l];
        if
        :: condition -> append(t, l, order, updated); inherit(l, newest[l])
        :: else
        fi
    }
}

/** The strong exchange: a failed one reads the newest write relaxed. */
inline compare_exchange(t, l, order, expected, desired, succeeded) {
    atomic {
        land_if_same_word(t, l);
        if
        :: newest_value(l) == expected ->
            merge(t, l, newest[l], order);
            append(t, l, order, desired);
            inherit(l, newest[l]);
            succeeded = true
        :: else ->
            expected = newest_value(l);
            current[thread_index(t, l)] = newest[l];
            merge_acquired(t, l, newest[l]);
            succeeded = false
        fi
    }
}

/** The no-return add: a plain read-modify-write, except the relaxed one under far memory. */
inline add_no_return(t, l, order, operand) {
    atomic {
#if memory == far
        if
        :: order == order_relaxed ->
            land_if_same_word(t, l);
            assert(pending_location[t] == 0);
            pending_location[t] = (l) + 1;
            pending_operand[t] = operand
        :: else ->
            land_if_same_word(t, l);
            summed = newest_value(l) + operand;
            append(t, l, order, summed);
            inherit(l, newest[l]);
            summed = 0
        fi
#else
        land_if_same_word(t, l);
        summed = newest_value(l) + operand;
        append(t, l, order, summed);
        inherit(l, newest[l]);
        summed = 0
#endif
    }
}

/** Merges what the thread's relaxed loads picked up into its @c current view. */
inline fence_acquire(t) {
    atomic {
        for (view_location : 0 .. location_count - 1) {
            if
            :: acquired[thread_index(t, view_location)] > current[thread_index(t, view_location)] ->
                current[thread_index(t, view_location)] = acquired[thread_index(t, view_location)]
            :: else
            fi
        };
        view_location = 0
    }
}

/** Captures the thread's @c current view as the one its later relaxed writes carry. */
inline fence_release(t) {
    atomic {
        for (view_location : 0 .. location_count - 1) {
            released[thread_index(t, view_location)] = current[thread_index(t, view_location)]
        };
        view_location = 0
    }
}

#endif
