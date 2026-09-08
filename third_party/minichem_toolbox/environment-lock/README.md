# Unified runtime lock files

`environment.yml` and `requirements.txt` are the portable source specifications. After a successful
bootstrap, `scripts/export_environment_lock.sh` writes the exact Linux conda artifact list,
installed pip versions, and critical runtime versions here.

`pip.lock.txt` contains only packages whose Conda metadata records `channel=pypi`; Conda packages are
already covered by the explicit lock and are not repeated as nonportable build-host `file://` URLs.
ASH is deliberately excluded from `pip.lock.txt` because it is installed from the unchanged
repository-local `vendor/ash-0.95` source. Its installed version and source policy are
recorded in `runtime.json`, together with SHA-256 hashes for the KNARR conversion implementation,
FairChem interface, and package definition. `MINICHEM_ASH_SOURCE` is an optional developer override;
normal reproduction does not require it.
