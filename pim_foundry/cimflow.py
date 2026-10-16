"""Dense INT8 lowering to CIMFlow's public SRAM ISA.

The upstream SystemC model executes timing/power, not CIM arithmetic.  A
separate instruction interpreter reads the *emitted* memory image and ISA and
checks lowering against an independent scalar matrix product.  These two
pieces of evidence must not be represented as native numerical execution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess

from .config import read_json

REVISION = "531abcb869b1b3af1f95cf923223f09e08725d96"
REPOSITORY = "https://github.com/BUAA-CI-LAB/CIMFlow-Simulator"
MAX_INSTRUCTIONS = 100_000
MAX_MEMORY_BYTES = 4 * 1024 * 1024


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _matrix(value, name):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not all(isinstance(row, (list, tuple)) and row for row in value):
        raise ValueError(f"{name} must have nonempty rows")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} has ragged rows")
    if any(type(x) is not int or not -128 <= x <= 127 for row in value for x in row):
        raise ValueError(f"{name} requires signed INT8 values")
    return [list(row) for row in value]


def _config(tile_k, tile_n, memory_size):
    """An explicit synthetic profile, not an extracted paper chip profile."""
    module = {"latency_cycle": 1, "static_power_mW": 0.01, "dynamic_power_mW": 0.01}
    return {
        "chip_config": {
            "core_cnt": 1,
            "core_config": {
                "cim_unit_config": {
                    "macro_total_cnt": 1, "macro_group_size": 1,
                    "macro_size": {"compartment_cnt_per_macro": tile_k,
                                   "element_cnt_per_compartment": tile_n,
                                   "row_cnt_per_element": 1, "bit_width_per_row": 8},
                    "bit_serial": True, "independent_mult": False,
                    "ipu": dict(module), "adder_tree": dict(module),
                    "shift_adder": dict(module), "result_adder": dict(module),
                    "sram": {"write_latency_cycle": 1, "read_latency_cycle": 1,
                             "static_power_mW": 0.01, "write_dynamic_power_per_bit_mW": 0.001,
                             "read_dynamic_power_per_bit_mW": 0.001},
                    "value_sparse": False, "bit_sparse": False, "input_bit_sparse": False,
                },
                "local_memory_unit_config": {"memory_list": [{
                    "name": "staging", "type": "ram", "hardware_config": {
                        "size_byte": memory_size, "width_byte": 16,
                        "read_latency_cycle": 1, "write_latency_cycle": 1,
                        "static_power_mW": 0.01, "read_dynamic_power_mW": 0.01,
                        "write_dynamic_power_mW": 0.01,
                        "has_image": True, "image_file": "memory.bin"}}]},
                "simd_unit_config": {"pipeline": True,
                    "functor_list": [{"name": "int32_add", "input_cnt": 2,
                        "data_bit_width": {"input1": 32, "input2": 32, "output": 32},
                        "functor_cnt": tile_n, "latency_cycle": 1,
                        "static_power_per_functor_mW": 0.01,
                        "dynamic_power_per_functor_mW": 0.01}],
                    "instruction_list": [{"name": "int32_add", "input_cnt": 2,
                        "opcode": "0x00", "input1_type": "vector", "input2_type": "vector",
                        "functor_binding_list": [{"input_bit_width": {"input1": 32, "input2": 32},
                                                  "functor_name": "int32_add"}]}]},
                "transfer_unit_config": {"pipeline": True},
            },
            "network_config": {"bus_width_byte": 16, "static_power_mW": 0.0,
                               "network_config_file_path": "network.json"},
            "address_space_config": [{"name": "cim_unit", "size": tile_k * tile_n},
                                     {"name": "staging", "size": memory_size}],
        },
        "sim_config": {"period_ns": 1.0, "sim_mode": "run_one_round", "data_mode": "real_data"},
    }


def compile_gemv(vector, weights, output: Path, *, tile_k=16, tile_n=16) -> dict:
    return compile_matmul([vector], weights, output, tile_k=tile_k, tile_n=tile_n)


def compile_matmul(inputs, weights, output: Path, *, tile_k=16, tile_n=16) -> dict:
    """Write native ISA, local SRAM image, geometry, and scalar expected output.

    Inputs are M x K, weights K x N, signed INT8; outputs are INT32.
    Ragged hardware tiles are zero padded. Local SRAM is prefilled; each
    weight tile is actually copied into CIM storage through native MEM_CPY.
    """
    a, b = _matrix(inputs, "inputs"), _matrix(weights, "weights")
    m, k, n = len(a), len(a[0]), len(b[0])
    if len(b) != k:
        raise ValueError("inputs K and weights K differ")
    if not (m <= 16 and k <= 128 and n <= 128):
        raise ValueError("native lowering supports M<=16, K<=128, N<=128")
    if any(type(t) is not int or t not in (8, 16, 32, 64) for t in (tile_k, tile_n)):
        raise ValueError("tile_k and tile_n must be one of 8,16,32,64")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    kp, np = math.ceil(k / tile_k) * tile_k, math.ceil(n / tile_n) * tile_n
    base = tile_k * tile_n
    memory = bytearray()

    def allocate(data):
        while len(memory) % 16:
            memory.append(0)
        addr = base + len(memory)
        memory.extend(data)
        return addr

    input_addr = allocate(bytes(x & 255 for row in a for x in row + [0] * (kp-k)))
    weight_addresses = {}
    for ks in range(0, kp, tile_k):
        for ns in range(0, np, tile_n):
            packed = bytes((b[i][j] if i < k and j < n else 0) & 255
                           for i in range(ks, ks+tile_k) for j in range(ns, ns+tile_n))
            weight_addresses[(ks, ns)] = allocate(packed)
    mask_addr = allocate(bytes([255] * (tile_n // 8)))
    partial_addr = allocate(bytes(tile_n * 4))
    output_addr = allocate(bytes(m * np * 4))
    memory.extend(bytes((-len(memory)) % 16))
    if len(memory) > MAX_MEMORY_BYTES:
        raise ValueError("local SRAM capacity exceeded")
    code = []

    def imm(reg, val, special=False):
        code.append({"opcode": 45 if special else 44, "rd": reg, "imm": val})

    for reg, val in ((0, 8), (1, 32), (2, 8), (4, 1), (6, 0), (16, 32), (17, 32), (20, 32)):
        imm(reg, val, True)
    imm(0, 0)
    imm(1, mask_addr)
    code.append({"opcode": 4, "rs": 0, "rt": 1, "GRP_B": True})
    for row in range(m):
        for ns in range(0, np, tile_n):
            for ks in range(0, kp, tile_k):
                imm(0, weight_addresses[(ks, ns)])
                imm(1, 0)
                imm(2, base)
                code.append({"opcode": 48, "rs": 0, "rd": 1, "rt": 2, "imm": 0})
                imm(0, input_addr + row * kp + ks)
                imm(1, tile_k)
                imm(2, 0)
                code.append({"opcode": 0, "rs": 0, "rt": 1, "re": 2, "GRP": True})
                imm(0, tile_n)
                imm(1, 0)
                imm(2, partial_addr)
                code.append({"opcode": 8, "rs": 0, "rt": 1, "rd": 2})
                imm(0, partial_addr)
                imm(1, output_addr + (row * np + ns) * 4)
                imm(2, tile_n)
                code.append({"opcode": 20, "rs": 0, "rt": 1, "rd": 1, "re": 2, "funct": 0})
    if len(code) > MAX_INSTRUCTIONS:
        raise ValueError("instruction budget exceeded")
    expected = [[sum(a[i][q] * b[q][j] for q in range(k)) for j in range(n)] for i in range(m)]
    manifest = {
        "schema": 1, "backend": "cimflow-sram", "source_revision": REVISION,
        "shape": {"m": m, "k": k, "n": n}, "padded_shape": {"k": kp, "n": np},
        "dtype": {"input": "int8", "weight": "int8", "accumulator": "int32", "output": "int32"},
        "tile": {"k": tile_k, "n": tile_n},
        "memory": {"cim_base": 0, "cim_size": base, "staging_base": base,
                   "staging_size": len(memory), "input_base": input_addr,
                   "output_base": output_addr, "output_row_stride": np * 4,
                   "weight_layout": "tile_row_major_signed_int8", "byte_order": "little"},
        "instruction_count": len(code), "cim_mvm_count": m * (kp // tile_k) * (np // tile_n),
        "weight_programming_bytes": m * kp * np,
        "timing_scope": "core_local_prefilled_including_cim_programming_and_k_reduction",
        "excluded_costs": ["host_to_staging", "output_to_host", "compiler_host_execution"],
        "profile_provenance": "synthetic_configurable_SRAM_profile_not_paper_chip_calibration",
        "native_numerical": False,
        "functional_semantics": "foundry_dense_signed_int8_CIM_ISA_extension; upstream_has_no_weight_arithmetic",
        "expected_output": expected,
    }
    output.mkdir(parents=True)
    (output / "memory.bin").write_bytes(memory)
    _write_json(output / "config.json", _config(tile_k, tile_n, len(memory)))
    _write_json(output / "network.json", {"latency": {}, "energy": {}})
    _write_json(output / "instructions.json", {"0": code, "metadata": {"op_count": 2*m*k*n}})
    _write_json(output / "workload.json", {"inputs": a, "weights": b})
    manifest["files"] = {name: _hash(output / name) for name in
                         ("memory.bin", "config.json", "network.json", "instructions.json", "workload.json")}
    _write_json(output / "manifest.json", manifest)
    return manifest


def verify(artifact: Path) -> dict:
    artifact = Path(artifact)
    manifest = read_json(artifact / "manifest.json")
    if manifest.get("schema") != 1 or manifest.get("source_revision") != REVISION:
        raise ValueError("unsupported CIMFlow artifact")
    required = {"memory.bin", "config.json", "network.json", "instructions.json", "workload.json"}
    if set(manifest.get("files", {})) != required:
        raise ValueError("artifact manifest has incorrect coverage")
    for name, expected in manifest["files"].items():
        if _hash(artifact / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    return manifest


def interpret(artifact: Path, *, verify_hashes=True) -> dict:
    """Execute emitted instructions and bytes, without reading expected values.

    This is the explicitly declared dense signed-INT8 functional ISA model.
    It supplies arithmetic absent from upstream; it is not upstream emulation.
    """
    artifact = Path(artifact)
    manifest = verify(artifact) if verify_hashes else read_json(artifact / "manifest.json")
    tile_k, tile_n = manifest["tile"]["k"], manifest["tile"]["n"]
    mem = manifest["memory"]
    if any(type(t) is not int or t not in (8,16,32,64) for t in (tile_k,tile_n)):
        raise ValueError("invalid artifact tile geometry")
    shape = manifest["shape"]
    if set(shape) != {"m","k","n"} or any(type(shape[key]) is not int or not 0 < shape[key] <= limit
                                           for key,limit in (("m",16),("k",128),("n",128))):
        raise ValueError("invalid artifact tensor shape")
    if (mem["cim_base"],mem["cim_size"],mem["staging_base"]) != (0,tile_k*tile_n,tile_k*tile_n):
        raise ValueError("invalid artifact memory geometry")
    memory_path = artifact/"memory.bin"
    if memory_path.stat().st_size > MAX_MEMORY_BYTES:
        raise ValueError("interpreter memory budget exceeded")
    if mem["staging_size"] != memory_path.stat().st_size:
        raise ValueError("artifact memory size mismatch")
    data = bytearray(mem["cim_size"]) + bytearray(memory_path.read_bytes())
    code = read_json(artifact / "instructions.json")["0"]
    if len(code) > MAX_INSTRUCTIONS:
        raise ValueError("interpreter instruction budget exceeded")
    general, special = [0]*32, [0]*32
    accumulator = None
    mask = [True]*tile_n
    counts = {}

    def bounds(addr, length):
        if type(addr) is not int or type(length) is not int or addr < 0 or length < 0 or addr + length > len(data):
            raise ValueError("ISA memory access out of bounds")

    def load(addr, length):
        bounds(addr, length)
        return data[addr:addr+length]

    def store(addr, values):
        bounds(addr, len(values))
        data[addr:addr+len(values)] = values

    def reg(ins, key):
        index = ins[key]
        if type(index) is not int or not 0 <= index < 32:
            raise ValueError("invalid ISA register")
        return general[index]

    for ins in code:
        op = ins["opcode"]
        counts[str(op)] = counts.get(str(op), 0)+1
        if op in (44, 45):
            index = ins["rd"]
            if type(index) is not int or not 0 <= index < 32 or type(ins["imm"]) is not int:
                raise ValueError("invalid immediate instruction")
            (special if op == 45 else general)[index] = ins["imm"]
        elif op == 48:
            store(reg(ins, "rd"), load(reg(ins, "rs"), reg(ins, "rt")))
        elif op == 4:
            if ins.get("GRP_B") is not True:
                raise ValueError("only broadcast CIM mask supported")
            bits = load(reg(ins, "rt"), tile_n//8)
            mask = [bool(bits[j//8] & (1 << (j%8))) for j in range(tile_n)]
        elif op == 0:
            if (special[0], special[1], special[2], special[4], reg(ins, "rt"), reg(ins, "re")) != (8,32,8,1,tile_k,0):
                raise ValueError("unsupported CIM precision, geometry, or row")
            x = struct.unpack(f"{tile_k}b", load(reg(ins, "rs"), tile_k))
            w = struct.unpack(f"{tile_k*tile_n}b", load(0, tile_k*tile_n))
            accumulator = [sum(x[i]*w[i*tile_n+j] for i in range(tile_k)) if mask[j] else 0
                           for j in range(tile_n)]
        elif op == 8:
            if accumulator is None or reg(ins, "rs") != tile_n or ins.get("OSUM") or ins.get("OSUM_MOV"):
                raise ValueError("CIM_OUT without compatible accumulator")
            store(reg(ins, "rd"), struct.pack(f"<{tile_n}i", *accumulator))
        elif op == 20:
            if (special[16],special[17],special[20],ins["funct"]) != (32,32,32,0):
                raise ValueError("unsupported SIMD arithmetic")
            count = reg(ins, "re")
            if not 0 < count <= tile_n:
                raise ValueError("invalid SIMD length")
            left = struct.unpack(f"<{count}i", load(reg(ins,"rs"), count*4))
            right = struct.unpack(f"<{count}i", load(reg(ins,"rt"), count*4))
            try:
                values = struct.pack(f"<{count}i", *(x+y for x,y in zip(left,right)))
            except struct.error as exc:
                raise ValueError("INT32 accumulation overflow") from exc
            store(reg(ins,"rd"), values)
        else:
            raise ValueError(f"unsupported functional ISA opcode: {op}")
    output = [list(struct.unpack(f"<{shape['n']}i", load(mem["output_base"]+i*mem["output_row_stride"], shape["n"]*4)))
              for i in range(shape["m"])]
    return {"output": output, "opcode_counts": counts,
            "output_bytes_sha256": hashlib.sha256(b"".join(struct.pack(f"<{shape['n']}i",*row) for row in output)).hexdigest()}


def execute(artifact: Path, simulator: Path, *, timeout=60.0) -> dict:
    """Run the native binary and independently check compiled ISA arithmetic."""
    artifact, simulator = Path(artifact).resolve(), Path(simulator).resolve()
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("timeout must be in (0,300]")
    manifest = verify(artifact)
    source = simulator.parent.parent
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "diff", "--name-only", "HEAD", "--"],
                           check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if revision != REVISION or dirty:
        raise ValueError("CIMFlow execution requires the unmodified pinned source checkout")
    functional = interpret(artifact)
    workload = read_json(artifact / "workload.json")
    a,b = _matrix(workload["inputs"],"inputs"), _matrix(workload["weights"],"weights")
    expected = [[sum(a[i][q]*b[q][j] for q in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
    if functional["output"] != expected or expected != manifest["expected_output"]:
        raise ValueError("compiled ISA differs from independent matrix reference")
    report_path = artifact / "native-report.json"
    if report_path.exists() or (artifact / "execution.json").exists():
        raise FileExistsError("native execution artifact already exists")
    command = [str(simulator), str(artifact/"config.json"), str(artifact/"instructions.json"),
               "-o", str(report_path), "--format", "json"]
    try:
        result = subprocess.run(command, cwd=artifact, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (artifact/"stdout.txt").write_bytes(exc.stdout or b"")
        (artifact/"stderr.txt").write_bytes(exc.stderr or b"")
        raise RuntimeError("CIMFlow execution timed out") from exc
    (artifact/"stdout.txt").write_text(result.stdout)
    (artifact/"stderr.txt").write_text(result.stderr)
    if result.returncode or re.search(r"\bERROR\b", result.stdout + result.stderr):
        raise RuntimeError(f"CIMFlow execution failed; inspect {artifact/'stderr.txt'}")
    native = read_json(report_path)
    latency, energy = native.get("latency_"), native.get("total_energy_")
    if any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in (latency,energy)):
        raise ValueError("invalid native timing or energy report")
    if native.get("op_count_") != 2*manifest["shape"]["m"]*manifest["shape"]["k"]*manifest["shape"]["n"]:
        raise ValueError("native report operation count mismatch")
    report = {"schema":1,"backend":"cimflow-sram","source_revision":REVISION,
              "evidence_kind":"paired_instruction_functional_and_native_timing",
              "lowering_verified":True,"native_numerical":False,"hardware_calibrated":False,
              "output":functional["output"],"functional":functional,
              "native_timing_ns":latency*1_000_000,"native_energy_pj":energy,
              "timing_scope":manifest["timing_scope"], "excluded_costs":manifest["excluded_costs"],
              "profile_provenance":manifest["profile_provenance"],
              "command":command,"returncode":result.returncode,
              "source_clean":True,
              "binary_sha256":_hash(simulator),"manifest_sha256":_hash(artifact/"manifest.json"),
              "runtime_libraries":{str(path.relative_to(simulator.parent)):_hash(path) for path in
                                   simulator.parent.rglob("*.so.*") if not path.is_symlink()},
              "raw_files":{name:_hash(artifact/name) for name in ("native-report.json","stdout.txt","stderr.txt")}}
    _write_json(artifact/"execution.json",report)
    return report
