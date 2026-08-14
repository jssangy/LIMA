execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${SRC}
    OUTPUT_VARIABLE hash OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT hash)
    set(hash "unknown")
endif()
# Only tracked changes can alter the compiled program.  Ignore unrelated
# untracked experiment artifacts so the embedded provenance is not marked
# dirty when the source at HEAD is byte-identical.
execute_process(COMMAND git status --porcelain --untracked-files=no
    WORKING_DIRECTORY ${SRC}
    OUTPUT_VARIABLE dirty ERROR_QUIET)
if(dirty)
    set(hash "${hash}-dirty")
endif()
set(content "#pragma once\n#define LIMA_COMMIT \"${hash}\"\n")
if(EXISTS ${OUT})
    file(READ ${OUT} old)
else()
    set(old "")
endif()
if(NOT old STREQUAL content)
    file(WRITE ${OUT} "${content}")
endif()
