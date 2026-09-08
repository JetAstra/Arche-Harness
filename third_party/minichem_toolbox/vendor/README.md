# Vendored Source

`ash-0.95/` is the ASH 0.95 runtime source snapshot used by the MiniChem
`ash_neb_uma` backend. Its computational modules are unchanged. It is source
code, not cache payload, and is versioned with the toolbox.

The snapshot excludes upstream helper scripts, ASH's own test suite, the
original build directory, generated metadata, bytecode, and the optional
precompiled `LJCoulombv1.so` MM accelerator. MiniChem's ASH/KNARR/FairChem NEB
path does not use those excluded files.

Critical source hashes are recorded in `environment-lock/runtime.json`.
Bootstrap copies this tree to a temporary build directory before installation,
so the vendored source remains unchanged.
