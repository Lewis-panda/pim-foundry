"""Private isolated worker; inspected upstream functions only, no setup script."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys


def main():
    request = json.load(sys.stdin)
    root = Path(request["upstream"])
    sys.path.insert(0, str(root / "src"))
    import generate_configs

    output = Path(request["output"])
    if request["action"] == "enumerate":
        with contextlib.redirect_stdout(io.StringIO()):
            path = generate_configs.generate_configs_3D(**request["workload"], output_dir=str(output))
        result = json.loads(Path(path).read_text())
    elif request["action"] == "compile":
        source = root / "src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py"
        spec = importlib.util.spec_from_file_location("dcc_hbm_gemv", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        trace = output / "kernel.trace"
        with contextlib.redirect_stdout(io.StringIO()):
            module.run_GEMV(request["tiling"], str(trace))
        result = {"trace": str(trace), "yaml": str(trace.with_suffix(".yaml"))}
    else:
        raise ValueError("unknown worker action")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
