# Custom Lookup Table

When your ONNX model uses **Sigmoid** (or other activations that use a built-in lookup table), you can optionally replace the default table with a **user-defined piecewise-linear (PWL)** lookup table. This is done by setting `custom_lookup_path` (Python: `py_run_args.custom_lookup_path`, CLI: `--custom-lookup-path`) to the path of a JSON file that describes the PWL mapping.

## How to produce the lookup table

The JSON file must contain three arrays:

| Field         | Length | Description |
|---------------|--------|-------------|
| `breakpoints` | n+1    | Strictly increasing x-values. Segment *i* is the interval `[breakpoints[i], breakpoints[i+1])`. |
| `slopes`      | n      | For segment *i*, the map is `y = slopes[i] * x + intercepts[i]`. |
| `intercepts`  | n      | Intercept for each segment. |

**Steps to generate the file:**

1. Choose breakpoints that cover the range of inputs your activation will see (e.g. for sigmoid, a symmetric range like `[-6, 6]`).
2. Evaluate the target function (e.g. `sigmoid(x) = 1/(1+exp(-x))`) at each breakpoint.
3. For each segment `[a, b]`:
   - `slope = (f(b) - f(a)) / (b - a)`
   - `intercept = f(a) - slope * a`
4. Write `breakpoints`, `slopes`, and `intercepts` to a JSON file.

An example for a 4-segment sigmoid approximation is in `examples/pwl_sigmoid_example.json`. The notebook `examples/notebooks/custom_lookup_demo.ipynb` shows how to generate this in Python and run the full EZKL pipeline with a custom lookup.

## Caveats and soundness

- **Correctness:** The PWL table is used both for the circuit (constraint system) and for witness generation. If the table does not match the function your model was trained with (e.g. a poor approximation of sigmoid), the committed output may differ from the true model output, which can lead to failed verification or unsound claims.
- **Range:** Inputs outside the breakpoint range are extrapolated using the first or last segment. Ensure your breakpoints cover the actual input range (see calibration) to avoid large approximation error.
- **Error handling:** Invalid JSON, non-increasing breakpoints, or mismatched array lengths will produce clear errors at load time (see `src/circuit/ops/lookup.rs`). The loader validates structure and reports soundness-related warnings where applicable.

## Usage

- **Python:** Set `py_run_args.custom_lookup_path = "/path/to/pwl.json"` (absolute path recommended) before calling `ezkl.gen_settings(...)`. The path is stored in the generated settings and used by `compile_circuit`, `gen_witness`, and proving.
- **CLI:** Pass `--custom-lookup-path /path/to/pwl.json` when running `gen-settings`. Use the same settings file for `calibrate-settings`, `compile-circuit`, and `prove`.
