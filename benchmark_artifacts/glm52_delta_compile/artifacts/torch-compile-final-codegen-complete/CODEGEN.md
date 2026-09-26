# Final word-based CPU stage: generated-code evidence

Frozen module SHA256: `281534dfa6b8fef9840f17af2522c3d8100097a68baf0b30bf0c2e1af69a7788`; standalone commit `6fb0e2a9d81fc343863ba6015512fe93992892e4`.

Text was retained from the exact C2 final replay cache after all timing completed. Collection only read source files and disassembled already-built libraries; it did not run or compile kernels. No binaries are included. `proof.json` records source and original library SHA256 values.

- The three generated C++ wrappers contain `py::gil_scoped_release_simple release;`.
- The retained generated Python modules contain the full generated C++ kernels and wrapper compile calls. `cbcajiukrcdieopikltodknnw4sf2rnqiib5hlrythekremhmwup.py` materializes XOR and the owned snapshot in the same loop, with 16 int32 words (64 bytes) per vector load. `c7v7ksfyhytxghbbkzs7tow7v3avlzxx5vivi64567xtiziqgtf4.py` shows the four byte masks, exact int32 row reduction, int64 conversion, and native ATen cumsum followed by per-name boundary extraction.
- The two preparation calls remain separate: counts and metadata first; owned XOR/snapshot materialization only when at least one named tensor changed. These are not one fused graph or one traversal.
- Generated sources contain no OpenMP team directives; the selected module requests one compiled kernel thread, and the encoder uses up to32 outer workers. This is distinct from the discarded8×4 prototype.
- Disassembly excerpts retain their function-symbol headers. The actual materialization kernel contains AVX-512 `vmovdqu64` and `vpxord`; the count kernels contain `vpandd`, `vpcmpneqd`, and `vpaddd`. The compiler calls request a vector ISA, and the built-library instructions establish AVX-512 use on this C2 host. No standalone compiler-command JSON files were present in this cache.
- Compiler counters and ISA evidence are not speedup evidence. The full paired replay reports the unchanged improvement and changed-update regression separately.

| Generated wrapper | GIL-release line | AVX-512 instruction lines in linked library |
| --- | ---: | ---: |
| [c2wawido3dtafcfncl27mn6bwebanaump7ohuzk3i7yggo7sw5je.main.cpp](c2wawido3dtafcfncl27mn6bwebanaump7ohuzk3i7yggo7sw5je.main.cpp) | 21 | 51 |
| [c76clulz2qgtjthxdgzq35qxoj4ofhmjfihkgfpldivebo3sm765.main.cpp](c76clulz2qgtjthxdgzq35qxoj4ofhmjfihkgfpldivebo3sm765.main.cpp) | 34 | 580 |
| [cxw6jn3k65kmhrieuem3rjnbovnqxluqmhsfjm4y57c2osnl34tm.main.cpp](cxw6jn3k65kmhrieuem3rjnbovnqxluqmhsfjm4y57c2osnl34tm.main.cpp) | 30 | 287 |
