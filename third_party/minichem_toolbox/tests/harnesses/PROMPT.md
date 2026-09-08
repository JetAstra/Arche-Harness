# MiniChem Three-Layer Interface Test

Work only inside the current workspace. Use the `minichem_toolbox` MCP server as the chemistry
execution interface. This is a small interface validation, not a literature-reproduction task.

Complete all three layers:

1. **Predefined Action:** inspect and execute an appropriate xTB Action for a water molecule using
   the explicit geometry below. Record the returned energy and artifacts.
2. **Native software:** inspect the Gaussian native interface, write a complete Gaussian 16 input,
   validate it, and run a B3LYP/6-31G(d) geometry optimization and frequency calculation for the same
   neutral singlet water molecule. Use at most 2 CPU cores and 2 GB memory. Poll and collect the job.
3. **Python program:** inspect an analysis runtime, write and submit a short Python program that uses
   cclib to parse the completed Gaussian output and writes `outputs/parsed_gaussian.json` through the
   managed `researchchem_job.JobContext` interface. Include the final electronic energy, optimized
   coordinates, vibrational frequencies, and whether Gaussian terminated normally.

Initial Cartesian geometry in angstrom:

```text
O   0.000000   0.000000   0.000000
H   0.758602   0.000000   0.504284
H  -0.758602   0.000000   0.504284
```

Write `report/report.md` with the exact methods, software versions when available, job statuses,
energies, frequency check, artifact paths, and a short comparison of xTB versus Gaussian. Do not
claim scientific agreement from this smoke test. Do not finish until all asynchronous jobs are
terminal and the report plus `outputs/parsed_gaussian.json` exist.
