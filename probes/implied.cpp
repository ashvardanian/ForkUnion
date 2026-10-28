/**
 *  @file probes/implied.cpp
 *  @author Ash Vardanian
 *  @date October 28, 2026
 *  @brief Whether the floor's predefined macros imply the macro @c FORKUNION_PROBE_MACRO names.
 */
#include "forkunion/types.hpp"
#if !FORKUNION_PROBE_MACRO
#error
#endif
