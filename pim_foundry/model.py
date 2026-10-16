"""A deliberately narrow kernel contract for the pinned DCC backend."""

from dataclasses import asdict, dataclass
from math import prod


@dataclass(frozen=True)
class Gemv:
    """Vector[B,H,K] times matrix[B,H,K,N] -> [B,H,N].

    FP16 is the upstream trace generator's element-size assumption only.
    This workbench accepts shapes, not tensor values.
    """

    batch: int = 1
    nhead: int = 16
    dhead: int = 64
    seq_len: int = 256

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or not 1 <= value <= 65536:
                raise ValueError(f"{name} must be an integer in [1, 65536]")
        if self.batch * self.nhead > 65536:
            raise ValueError("batch*nhead must be <= 65536 for bounded enumeration")

    @property
    def dimensions(self) -> tuple[int, int, int]:
        return self.batch * self.nhead, self.dhead, self.seq_len


def inspect_tiling(workload: Gemv, tiling: list) -> dict:
    """Conservative experimental subset; not a hardware legality proof."""
    if not isinstance(tiling, list) or len(tiling) != 3 or any(
        not isinstance(axis, list) or len(axis) != 3 or
        any(type(x) is not int or x <= 0 for x in axis) for axis in tiling
    ):
        raise ValueError("tiling must contain three [groups, cores, local_extent] lists")
    reasons = []
    for label, axis, size in zip(("heads", "input", "output"), tiling, workload.dimensions):
        if any(x & (x - 1) for x in axis[:2]):
            reasons.append(f"{label}: group/core factors must be powers of two")
        if prod(axis) != size:
            reasons.append(f"{label}: require exact coverage without padding")
    groups = prod(x[0] for x in tiling)
    cores = prod(x[1] for x in tiling)
    if groups > 16 or cores > 64:
        reasons.append("mapping exceeds the fixed 16 groups / 64 cores per group")
    heads, k, n = (x[2] for x in tiling)
    if heads != 1:
        reasons.append("local heads must equal 1: upstream return-address aliasing is unresolved")
    if not (16 <= k <= 128 and k % 16 == 0):
        reasons.append("local input extent must be a multiple of 16 in [16,128]: upstream reduction-loop limit")
    if n % 16:
        reasons.append("local output extent must be a multiple of 16")
    rounded = lambda x: (x + 31) // 32 * 32
    local_bytes = sum(rounded(x * 2) for x in (heads * k, heads * k * n, heads * n))
    if local_bytes > 2**24:
        reasons.append("requires upstream multi-iteration spilling, which is not verified")
    participants = groups * cores
    chunks_k, chunks_n = (k + 15) // 16, (n + 15) // 16
    output_blocks = (chunks_n + 7) // 8
    even_bank_bases = groups * ((cores + 1) // 2)
    odd_bank_bases = groups * (cores // 2)
    mac_bases = (even_bank_bases + 1) // 2 + (odd_bank_bases + 1) // 2
    # Resource bound derived from the inspected single-head generator. Neither
    # arithmetic work nor timing can be inferred from this command count.
    predicted_commands = (
        2 * chunks_k * participants + 32 * ((chunks_k + 15) // 16)
        + 2 * chunks_n * participants + output_blocks * chunks_k * participants
        + chunks_n * chunks_k * mac_bases
        + 16 + chunks_n * participants
    )
    if predicted_commands > 250000:
        reasons.append("exceeds this workbench's 250000-command generation budget")
    return {
        "eligible_for_trace_experiment": not reasons,
        "rejection_reasons": reasons,
        "groups_used": groups, "cores_per_group_used": cores,
        "local_bytes": local_bytes, "predicted_commands": predicted_commands,
        "coverage": [prod(axis) for axis in tiling],
    }
