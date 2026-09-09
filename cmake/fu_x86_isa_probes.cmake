# cmake/fu_x86_isa_probes.cmake - x86 capability probes
#
# Which of the x86 capability bits this toolchain can build. Probe source lives in probes/x86_*.cpp.
include(cmake/fu_isa_probe.cmake)

fu_isa_probes_begin_()
fu_isa_probe_(fu_target_x86_pause "probes/x86_pause.cpp")
fu_isa_probe_(fu_target_x86_tpause "probes/x86_tpause.cpp")
fu_isa_probe_(fu_target_x86_cldemote "probes/x86_cldemote.cpp")
fu_isa_probe_(fu_target_x86_cmpccxadd "probes/x86_cmpccxadd.cpp")
fu_isa_probe_(fu_target_x86_raoint "probes/x86_raoint.cpp")
fu_isa_probes_end_()

fu_build_isa_defs_(x86 "x86" "X86_PAUSE;X86_TPAUSE;X86_CLDEMOTE;X86_CMPCCXADD;X86_RAOINT")
