# Project Aura Engineering Rules

- Preserve existing completed functionality.
- Never rewrite notebooks 01–04 unnecessarily.
- Python 3.11 is the preferred environment.
- CPU execution must always be supported.
- GPU acceleration is optional.
- Use existing source modules instead of duplicating logic.
- Do not use arbitrary eval or exec.
- Never bypass agent permission checks.
- Execution success is not verification success.
- Never fabricate notebook outputs or test results.
- Run pytest after modifying source code.
- Validate notebooks before modifying or executing them.
- Do not download ML weights automatically during tests.
- Clearly distinguish synthetic data from real measurements.
- Do not commit or push without explicit approval.