# cmake/fu_arm64_isa_probes.cmake - Arm capability probes
#
# Which of the Arm capability bits this toolchain can build. Probe source lives in probes/arm64_*.cpp.
include(cmake/fu_isa_probe.cmake)

fu_isa_probes_begin_()
fu_isa_probe_(fu_target_arm64_yield "probes/arm64_yield.cpp")
fu_isa_probe_(fu_target_arm64_wfet "probes/arm64_wfet.cpp")
fu_isa_probe_(fu_target_arm64_dc_cvac "probes/arm64_dc_cvac.cpp")
fu_isa_probe_(fu_target_arm64_lse "probes/arm64_lse.cpp")
fu_isa_probe_(fu_target_arm64_rcpc "probes/arm64_rcpc.cpp")
fu_isa_probes_end_()

fu_build_isa_defs_(arm64 "Arm" "ARM64_YIELD;ARM64_WFET;ARM64_DC_CVAC;ARM64_LSE;ARM64_RCPC")
