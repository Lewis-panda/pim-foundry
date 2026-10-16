"""A specified SIMD16 HBM research ISA with an independent byte interpreter.

The functional model is Foundry's, not a claim about Samsung register semantics.
Its address-bearing PIM requests can be projected into the repaired DCC timing
model. Timing projection and functional instruction execution remain distinct.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import struct

from .config import read_json

PROFILE = "foundry-hbm-simd16-fp16-fp32/v1"
LANES = 16
BANK_BYTES = 16 * 1024 * 1024
CHANNEL_BYTES = 1024 * 1024 * 1024


def _round(value, fmt):
    try:
        result = struct.unpack("<" + fmt, struct.pack("<" + fmt, value))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("nonfinite or overflowing floating-point instruction result") from error
    if not math.isfinite(result):
        raise ValueError("nonfinite floating-point instruction result")
    return result


def _matrix(values, name):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(name + " must be a nonempty rectangular matrix")
    if any(not isinstance(row, (list, tuple)) or not row for row in values):
        raise ValueError(name + " must have nonempty rows")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ValueError(name + " must be rectangular")
    result = []
    for row in values:
        normalized = []
        for value in row:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(name + " values must be finite numbers")
            normalized.append(_round(value, "e"))
        result.append(normalized)
    return result


def _bank_address(index):
    return (index % 16) * CHANNEL_BYTES + (index // 16) * BANK_BYTES


def timing_projection(program):
    """Canonical conservative request stream bound to these exact instructions."""
    trace = []
    names = {"RESET": "PIM_ACC_RESET", "LOAD16": "PIM_LD_OP1", "MAC16": "PIM_MAC_OP1", "STORE16": "PIM_WB_ACC"}
    for instruction in program["instructions"]:
        address = _bank_address(instruction["bank"]) if instruction["op"] == "RESET" else instruction["address"]
        trace.append(f"{names[instruction['op']]} 0x{address:08x}")
        trace.append(f"PIM_BARRIER 0x{(address // CHANNEL_BYTES) * CHANNEL_BYTES:08x}")
    return "\n".join(trace) + "\n"


def compile_matmul(inputs, weights, output: Path) -> dict:
    """Lower FP16 matrices into memory bytes and SIMD instructions, with padding.

    Accumulation is sequential FP32 multiply then add, round-to-nearest-even;
    output is FP16. Every tile has private aligned vector/weight/output storage.
    Host initialization is excluded from the compute trace and reported in bytes.
    """
    inputs, weights = _matrix(inputs, "inputs"), _matrix(weights, "weights")
    m, k, n = len(inputs), len(weights), len(weights[0])
    if len(inputs[0]) != k:
        raise ValueError("matmul shape mismatch")
    if m * k * n > 2_000_000 or max(m, k, n) > 65536:
        raise ValueError("banked lowering exceeds the 2000000-MAC resource budget")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("output path already exists")
    memory, instructions, tiles = [], [], []
    used = [0] * 1024
    for row in range(m):
        for first in range(0, n, LANES):
            tile_index = len(tiles)
            bank = tile_index % 1024
            bank_base = _bank_address(bank)
            vector_base = bank_base + used[bank]
            weight_base = vector_base + k * 32
            result_base = weight_base + k * 32
            used[bank] += (2 * k + 1) * 32
            if used[bank] > BANK_BYTES:
                raise ValueError("packed data exceeds bank capacity")
            vector_bytes = b"".join(struct.pack("<16e", *([value] * LANES)) for value in inputs[row])
            weight_bytes = b"".join(struct.pack("<16e", *[
                weights[reduction][column] if column < n else 0.0
                for column in range(first, first + LANES)]) for reduction in range(k))
            for address, data, role in ((vector_base, vector_bytes, "input"),
                                        (weight_base, weight_bytes, "weights"),
                                        (result_base, bytes(32), "output")):
                memory.append({"address": address, "data": base64.b64encode(data).decode(),
                               "role": role, "length": len(data)})
            instructions.append({"op": "RESET", "bank": bank})
            for reduction in range(k):
                instructions.append({"op": "LOAD16", "bank": bank, "address": vector_base + reduction * 32})
                instructions.append({"op": "MAC16", "bank": bank, "address": weight_base + reduction * 32})
            instructions.append({"op": "STORE16", "bank": bank, "address": result_base})
            tiles.append({"row": row, "first_column": first, "valid_lanes": min(LANES, n - first),
                          "address": result_base, "bank": bank})
    program = {
        "schema_version": 1, "profile": PROFILE, "shape": [m, k, n],
        "input_dtype": "float16", "accumulator_dtype": "float32", "output_dtype": "float16",
        "arithmetic": "sequential non-fused FP32 multiply and add; FP16 store; nearest-even; nonfinite errors",
        "memory": memory, "instructions": instructions, "outputs": tiles,
        "packed_initialization_bytes": sum(item["length"] for item in memory if item["role"] != "output"),
        "reserved_bytes": sum(item["length"] for item in memory),
        "padded_output_bytes": len(tiles) * 32,
        "host_initialization_in_timing_trace": False,
        "native_hardware_semantics_claimed": False,
    }
    # A barrier after each instruction makes this deliberately conservative
    # projection honor this model's dependencies, independent of reordering.
    trace = timing_projection(program)
    output.mkdir(parents=True)
    (output / "program.json").write_text(json.dumps(program, separators=(",", ":"), allow_nan=False) + "\n")
    (output / "kernel.trace").write_text(trace)
    manifest = {"schema_version": 1, "profile": PROFILE,
                "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                          for name in ("program.json", "kernel.trace")}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {"profile": PROFILE, "shape": [m, k, n], "instructions": len(instructions),
            "trace_commands": len(trace.splitlines()), "packed_initialization_bytes": program["packed_initialization_bytes"],
            "reserved_bytes": program["reserved_bytes"], "output": str(output)}


def compile_gemv(vector, weights, output: Path):
    return compile_matmul([vector], weights, output)


def interpret_program(program: dict) -> list[list[float]]:
    """Execute emitted instructions on packed bytes, without original tensors."""
    if not isinstance(program, dict) or program.get("profile") != PROFILE:
        raise ValueError("unsupported banked instruction profile")
    shape = program.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(v) is not int or v <= 0 for v in shape) or math.prod(shape) > 2_000_000):
        raise ValueError("invalid banked shape or resource budget")
    m, k, n = shape
    blocks = {}
    output_blocks = set()
    for region in program["memory"]:
        address = region["address"]
        if type(address) is not int or address < 0 or address % 32 or address >= 16 * CHANNEL_BYTES:
            raise ValueError("memory region address must be an aligned in-range integer")
        try:
            data = base64.b64decode(region["data"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid memory bytes") from exc
        if not data or len(data) % 32 or len(data) != region["length"]:
            raise ValueError("memory region length mismatch")
        if address // BANK_BYTES != (address + len(data) - 1) // BANK_BYTES:
            raise ValueError("memory region crosses a bank boundary")
        for offset in range(0, len(data), 32):
            current = address + offset
            if current in blocks:
                raise ValueError("overlapping memory regions")
            blocks[current] = data[offset:offset + 32]
            if region["role"] == "output":
                output_blocks.add(current)
    accumulators, operands, written = {}, {}, set()
    instructions = program["instructions"]
    if not isinstance(instructions, list) or len(instructions) > 4_500_000:
        raise ValueError("invalid instruction count")
    for instruction in instructions:
        op, bank = instruction["op"], instruction["bank"]
        if set(instruction) != ({"op", "bank"} if op == "RESET" else {"op", "bank", "address"}):
            raise ValueError("invalid instruction fields")
        if type(bank) is not int or not 0 <= bank < 1024:
            raise ValueError("instruction bank out of range")
        if op == "RESET":
            accumulators[bank] = [0.0] * LANES
            operands.pop(bank, None)
            continue
        address = instruction.get("address")
        base = _bank_address(bank)
        if type(address) is not int or address not in blocks or not base <= address < base + BANK_BYTES:
            raise ValueError("instruction addresses unallocated memory or another bank")
        if bank not in accumulators:
            raise ValueError("instruction used an uninitialized accumulator")
        if op == "LOAD16":
            if address in output_blocks:
                raise ValueError("operand loads cannot read uninitialized outputs")
            operands[bank] = struct.unpack("<16e", blocks[address])
        elif op == "MAC16":
            if bank not in operands or address in output_blocks:
                raise ValueError("MAC requires a loaded operand and initialized weights")
            weights = struct.unpack("<16e", blocks[address])
            for lane in range(LANES):
                product = _round(operands[bank][lane] * weights[lane], "f")
                accumulators[bank][lane] = _round(accumulators[bank][lane] + product, "f")
        elif op == "STORE16":
            if address not in output_blocks or address in written:
                raise ValueError("writeback aliases inputs or an existing result")
            values = [_round(value, "e") for value in accumulators[bank]]
            blocks[address] = struct.pack("<16e", *values)
            written.add(address)
        else:
            raise ValueError("unsupported instruction: " + str(op))
    result = [[None] * n for _ in range(m)]
    consumed_output_addresses = set()
    for tile in program["outputs"]:
        address, row, first, count = (tile[key] for key in ("address", "row", "first_column", "valid_lanes"))
        bank = tile.get("bank")
        if (any(type(value) is not int for value in (address, row, first, count, bank))
                or not 0 <= bank < 1024 or address in consumed_output_addresses
                or not _bank_address(bank) <= address < _bank_address(bank) + BANK_BYTES
                or address not in written or not 0 <= row < m or not 0 <= first < n
                or count != min(LANES, n - first)):
            raise ValueError("invalid output tile or unwritten output")
        consumed_output_addresses.add(address)
        values = struct.unpack("<16e", blocks[address])
        for lane in range(count):
            if result[row][first + lane] is not None:
                raise ValueError("duplicate output coverage")
            result[row][first + lane] = values[lane]
    if any(value is None for row in result for value in row):
        raise ValueError("incomplete output coverage")
    return result


def execute_functional(artifact: Path) -> dict:
    artifact = Path(artifact)
    manifest = read_json(artifact / "manifest.json")
    if manifest.get("profile") != PROFILE or set(manifest.get("files", {})) != {"program.json", "kernel.trace"}:
        raise ValueError("invalid banked manifest")
    for name, expected in manifest["files"].items():
        if hashlib.sha256((artifact / name).read_bytes()).hexdigest() != expected:
            raise ValueError("banked artifact hash mismatch: " + name)
    program = read_json(artifact / "program.json")
    values = interpret_program(program)
    if (artifact / "kernel.trace").read_text() != timing_projection(program):
        raise ValueError("timing trace differs from the functional instruction projection")
    return {"outputs": values, "profile": PROFILE,
            "functional_evidence": "instruction_interpreter_on_emitted_memory_bytes",
            "native_arithmetic": False, "reserved_bytes": program["reserved_bytes"],
            "packed_initialization_bytes": program["packed_initialization_bytes"],
            "native_timing_projection": "DCC requests; dependency barriers; explicit completion assumptions required"}
