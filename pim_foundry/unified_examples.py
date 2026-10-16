"""Versioned typed demonstrations and explicit native/software platform choices."""

from itertools import permutations
from pathlib import Path

from .typed import Arithmetic, Graph, Node, Tensor


def platform(kinds=("sram", "hbm", "gpu", "npu"), *, root=None):
    if not kinds or len(kinds) != len(set(kinds)) or any(kind not in ("sram", "hbm", "gpu", "npu") for kind in kinds):
        raise ValueError("choose unique sram,hbm,gpu,npu device kinds")
    default_root = Path.cwd() if (Path.cwd() / "pim_foundry").is_dir() else Path(__file__).resolve().parents[1]
    root = Path(root or default_root).resolve()
    devices = []
    for kind in kinds:
        if kind == "sram":
            backend = "cimflow-sram"
            options = {"executable": str(root / "external/cimflow-simulator/build/cim-simulator"),
                       "packing_bytes_per_ns": 32, "packing_energy_pj_per_byte": 0.1}
        elif kind == "hbm":
            backend = "hbm-simd16"
            options = {"runtime_descriptor": str(root / "runs/dcc-patched-final/runtime.json"),
                       "packing_bytes_per_ns": 32, "packing_energy_pj_per_byte": 0.1,
                       "native_energy_pj_per_cycle": 1.0}
        else:
            backend = "analytical"
            options = {"macs_per_ns": 16, "elements_per_ns": 16, "local_bytes_per_ns": 64,
                       "launch_ns": 10, "mac_energy_pj": 1.0, "element_energy_pj": 0.1,
                       "local_energy_pj_per_byte": 0.1}
        devices.append({"name": kind, "kind": kind, "backend": backend,
                        "capacity_bytes": 16 * 1024 * 1024, "idle_power_w": 0.01, "options": options})
    links = [{"source": source, "destination": destination, "bytes_per_ns": 16,
              "latency_ns": 5, "energy_pj_per_byte": 0.2}
             for source, destination in permutations(("host", *kinds), 2)]
    return {"schema_version": 1, "devices": devices, "links": links}


def workload(name="mixed"):
    fp = Arithmetic("float32")
    if name == "mixed":
        m, k, middle, n = 2, 7, 19, 11
        tensors = (
            Tensor("x", (m, k), "int8"),
            Tensor("sram_weights", (k, middle), "int8", tuple((i * 3 + 1) % 5 - 2 for i in range(k * middle))),
            Tensor("integer_hidden", (m, middle), "int32"),
            Tensor("fp_hidden", (m, middle), "float16"),
            Tensor("hbm_weights", (middle, n), "float16", tuple(((i * 5) % 7 - 3) / 4 for i in range(middle * n))),
            Tensor("hbm_output", (m, n), "float16"),
            Tensor("activated", (m, n), "float16"),
            Tensor("bias", (m, n), "float16", tuple((i % 3 - 1) / 4 for i in range(m * n))),
            Tensor("y", (m, n), "float16"),
        )
        nodes = (Node("sram_linear", "matmul", ("x", "sram_weights"), "integer_hidden"),
                 Node("convert", "cast", ("integer_hidden",), "fp_hidden", fp),
                 Node("hbm_linear", "matmul", ("fp_hidden", "hbm_weights"), "hbm_output", fp),
                 Node("activation", "relu", ("hbm_output",), "activated", fp),
                 Node("residual", "add", ("activated", "bias"), "y", fp))
        placement = {"sram_linear": "sram", "convert": "gpu", "hbm_linear": "hbm", "activation": "npu", "residual": "gpu"}
        inputs = {"x": tuple((i * 5 + 2) % 7 - 3 for i in range(m * k))}
    elif name in ("hbm-gpu", "hbm-npu"):
        accelerator = name.split("-")[1]
        m, k, n = 2, 17, 19
        tensors = (Tensor("x", (m, k), "float16"),
                   Tensor("w", (k, n), "float16", tuple((i % 7 - 3) / 8 for i in range(k * n))),
                   Tensor("hidden", (m, n), "float16"), Tensor("y", (m, n), "float16"))
        nodes = (Node("linear", "matmul", ("x", "w"), "hidden", fp),
                 Node("activation", "relu", ("hidden",), "y", fp))
        inputs = {"x": tuple((i % 5 - 2) / 4 for i in range(m * k))}
        placement = {"linear": "hbm", "activation": accelerator}
    else:
        raise ValueError("workload must be mixed, hbm-gpu or hbm-npu")
    graph = Graph(tensors, nodes, ("x",), ("y",))
    graph.validate()
    return graph, inputs, placement
