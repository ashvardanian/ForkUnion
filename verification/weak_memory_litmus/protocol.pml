/**
 *  @file verification/weak_memory_litmus/protocol.pml
 *  @author Ash Vardanian
 *  @date September 11, 2026
 *  @brief Calibration of `weak_memory.pml`: the classic shapes, one scenario each, each asserting
 *      the outcome the C++ model forbids.
 *
 *  A shape whose outcome a memory model admits fails its assertion, and every scenario beside this
 *  file expects exactly the failures RC11 would produce. The locations are @c data and @c flag: a
 *  message is the data written before the flag is raised.
 */
#include "../weak_memory.pml"

/** The words. */
#define data 0
#define flag 1

/** Writes the data relaxed, then raises the flag with @p flag_order. */
inline send(t, flag_order) {
    store(t, data, order_relaxed, 1);
    store(t, flag, flag_order, 1)
}

/** Reads the flag with @p flag_order, then the data with @p data_order: a reader that saw the flag
 *  at @p raised must see the data written. */
inline receive(t, flag_order, data_order, raised) {
    int seen_flag, seen_data;
    load(t, flag, flag_order, seen_flag);
    load(t, data, data_order, seen_data);
    assert(!(seen_flag == raised && seen_data == 0))
}
