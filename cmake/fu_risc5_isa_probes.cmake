# cmake/fu_risc5_isa_probes.cmake - RISC-V capability probes
#
# Which of the RISC-V capability bits this toolchain can build. Probe source lives in probes/risc5_*.cpp.
include(cmake/fu_isa_probe.cmake)

fu_instruction_set_probe_(fu_target_risc5_pause "probes/risc5_pause.cpp")
fu_instruction_set_probe_(fu_target_risc5_wrs "probes/risc5_wrs.cpp")
fu_instruction_set_probe_(fu_target_risc5_zicbom "probes/risc5_zicbom.cpp")
fu_instruction_set_probe_(fu_target_risc5_atomic "probes/risc5_atomic.cpp")
fu_instruction_set_probe_(fu_target_risc5_zacas "probes/risc5_zacas.cpp")

fu_build_instruction_set_definitions_("RISC-V" "RISC5_PAUSE;RISC5_WRS;RISC5_ZICBOM;RISC5_ATOMIC;RISC5_ZACAS")
