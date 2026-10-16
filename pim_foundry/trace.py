"""Streaming syntax and address observations for the pinned DCC HBM GEMV trace.

The geometry below is the geometry encoded by DCC's
``src/hbmpim_trace_gen/gen_trace_HBMPIM_GEMV.py``.  These observations do not
validate execution semantics, compute numerical results, or estimate latency.
"""

from __future__ import annotations

from pathlib import Path
import re


PREFETCH_BYTES = 32
BANK_STRIDE_BYTES = 16 * 2**20
CHANNEL_STRIDE_BYTES = 2**30
CHANNEL_COUNT = 16
ADDRESS_LIMIT_BYTES = CHANNEL_COUNT * CHANNEL_STRIDE_BYTES

OPCODES = (
    "LD",
    "ST",
    "PIM_BARRIER",
    "PIM_ACC_RESET",
    "PIM_LD_OP1",
    "PIM_MAC_OP1",
    "PIM_WB_ACC",
)

# Actual generator lines are shorter than 32 bytes.  A fixed bound also keeps
# malformed input from forcing an arbitrarily large allocation in readline().
_MAX_LINE_BYTES = 256
_HEX_ADDRESS = re.compile(r"0x[0-9a-fA-F]+\Z")


def analyze_trace(path: Path) -> dict:
    """Validate a trace and return JSON-serializable, deterministic observations.

    Each nonempty line must contain an allowed opcode and a hexadecimal address
    with a ``0x`` prefix.  Blank lines, comments, extra fields, non-ASCII text,
    oversized lines, and out-of-range addresses raise ``ValueError`` identifying
    the input line.  I/O errors propagate normally.  Memory use does not grow
    with the number of trace lines.

    Addresses on every command, including barriers, contribute to min/max,
    channel coverage, and alignment counts.  Alignment is reported, not required:
    the upstream generator can emit sub-prefetch offsets for ``PIM_LD_OP1``.
    """
    counts = {opcode: 0 for opcode in OPCODES}
    unaligned_counts = {opcode: 0 for opcode in OPCODES}
    channels: set[int] = set()
    non_barrier_channels: set[int] = set()
    instruction_count = 0
    min_address: int | None = None
    max_address: int | None = None

    with Path(path).open("rb") as source:
        line_number = 0
        while True:
            raw = source.readline(_MAX_LINE_BYTES + 1)
            if not raw:
                break
            line_number += 1
            prefix = f"line {line_number}: "
            if len(raw) > _MAX_LINE_BYTES:
                raise ValueError(prefix + f"trace line exceeds {_MAX_LINE_BYTES} bytes")
            try:
                line = raw.decode("ascii")
            except UnicodeDecodeError as error:
                raise ValueError(prefix + "trace line must contain ASCII text") from error
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(prefix + "expected exactly OP 0xADDR")
            opcode, address_text = fields
            if opcode not in counts:
                raise ValueError(prefix + f"unsupported opcode {opcode!r}")
            if _HEX_ADDRESS.fullmatch(address_text) is None:
                raise ValueError(prefix + f"invalid hexadecimal address {address_text!r}")
            address = int(address_text, 16)
            if address >= ADDRESS_LIMIT_BYTES:
                raise ValueError(
                    prefix + f"address must be in [0, {ADDRESS_LIMIT_BYTES})"
                )

            instruction_count += 1
            counts[opcode] += 1
            min_address = address if min_address is None else min(min_address, address)
            max_address = address if max_address is None else max(max_address, address)
            channel = address // CHANNEL_STRIDE_BYTES
            channels.add(channel)
            if opcode != "PIM_BARRIER":
                non_barrier_channels.add(channel)
            if address % PREFETCH_BYTES:
                unaligned_counts[opcode] += 1

    if instruction_count == 0:
        raise ValueError("line 1: empty trace")

    return {
        "instruction_count": instruction_count,
        "opcode_counts": counts,
        "host_command_count": counts["LD"] + counts["ST"],
        "barrier_count": counts["PIM_BARRIER"],
        "min_address": min_address,
        "max_address": max_address,
        "channels_touched": sorted(channels),
        "non_barrier_channels_touched": sorted(non_barrier_channels),
        "unaligned_address_count": sum(unaligned_counts.values()),
        "unaligned_address_counts_by_opcode": unaligned_counts,
        "channel_stride_bytes": CHANNEL_STRIDE_BYTES,
        "address_limit_bytes": ADDRESS_LIMIT_BYTES,
        "alignment_bytes": PREFETCH_BYTES,
        "observed_only": True,
        "notes": [
            "Counts and addresses are observations of the trace text only; "
            "no timing, numerical correctness, or execution validity is inferred.",
            "One PIM_MAC_OP1 command is not an established arithmetic operation count.",
            "host_command_count counts LD and ST commands, not transferred bytes; "
            "the trace protocol does not establish a byte count here.",
            "Address bounds, min/max, channels_touched, and alignment counts include "
            "barrier addresses; non_barrier_channels_touched excludes barriers.",
            "Unaligned addresses are reported without rejection; the pinned "
            "generator can produce sub-32-byte operand offsets.",
        ],
    }
