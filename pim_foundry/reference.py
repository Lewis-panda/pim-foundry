"""Tiled functional execution for illustrative SRAM/HBM/GPU/NPU logical devices.

Events describe actual work performed here, not vendor instructions or measured
timing. Capacity counts logical int64 storage; Python object overhead and exact
integer intermediate widths are outside this reference resource model.
"""

from .core import KernelResult, ReferenceDevice, Tensor, checked_integer


MAX_KERNEL_EVENTS = 100_000


def product_event_count(rows: int, k_size: int, n_size: int, device: ReferenceDevice) -> int:
    """Exact load-plus-compute count for validated product dimensions."""
    k_tiles = (k_size + device.tile_k - 1) // device.tile_k
    n_tiles = (n_size + device.tile_n - 1) // device.tile_n
    return 2 * rows * k_tiles * n_tiles


def check_event_budget(count: int) -> None:
    """Reject oversized reference traces before allocating any event objects."""
    if count > MAX_KERNEL_EVENTS:
        raise ValueError(f"reference execution exceeds the {MAX_KERNEL_EVENTS}-kernel-event limit")


_MATVEC_EVENTS = {
    "sram": "array_mvm",
    "hbm": "bank_mac",
    "gpu": "gpu_tiled_matvec",
    "npu": "npu_tiled_matvec",
}
_MATMUL_EVENTS = {
    "sram": "array_mvm",
    "hbm": "bank_mac",
    "gpu": "gpu_tiled_matmul",
    "npu": "npu_tiled_matmul",
}


def _validate_vector(x: tuple[int, ...], device: ReferenceDevice) -> None:
    if type(x) is not tuple or not x or len(x) > 1_000_000:
        raise ValueError("reference input must be a nonempty tuple of at most 1000000 elements")
    for value in x:
        checked_integer(value)
    if not isinstance(device, ReferenceDevice):
        raise ValueError("reference kernel requires a ReferenceDevice")


def _check_capacity(required_bytes: int, device: ReferenceDevice) -> None:
    if required_bytes > device.capacity_bytes:
        raise ValueError(
            f"{device.name}: kernel requires {required_bytes} logical bytes; "
            f"capacity is {device.capacity_bytes}"
        )


def _evaluate_product(
    x: tuple[int, ...], weights: Tensor, device: ReferenceDevice, rows: int, matrix: bool
) -> KernelResult:
    """One sequential input row at a time; each row streams all weight tiles."""
    k_size, n_size = weights.shape
    check_event_budget(product_event_count(rows, k_size, n_size, device))
    scratch = min(k_size, device.tile_k) * min(n_size, device.tile_n)
    _check_capacity(8 * (rows * (k_size + n_size) + scratch), device)

    output = [0] * (rows * n_size)
    events = []
    compute_kind = (_MATMUL_EVENTS if matrix else _MATVEC_EVENTS)[device.kind]
    for m in range(rows):
        for n0 in range(0, n_size, device.tile_n):
            n1 = min(n0 + device.tile_n, n_size)
            unit = (n0 // device.tile_n) % device.units
            for k0 in range(0, k_size, device.tile_k):
                k1 = min(k0 + device.tile_k, k_size)
                tile = tuple(
                    weights.data[k * n_size + n]
                    for k in range(k0, k1)
                    for n in range(n0, n1)
                )
                coordinates = {"k0": k0, "k1": k1, "n0": n0, "n1": n1, "unit": unit}
                if matrix:
                    coordinates["m"] = m
                events.append({"kind": "weight_load", **coordinates, "logical_bytes": 8 * len(tile)})
                for k in range(k0, k1):
                    for n in range(n0, n1):
                        output[m * n_size + n] += x[m * k_size + k] * tile[(k - k0) * (n1 - n0) + n - n0]
                events.append({"kind": compute_kind, **coordinates, "macs": len(tile)})
    return KernelResult(tuple(checked_integer(value) for value in output), tuple(events))


def _validate_product(x, weights, device, operation):
    _validate_vector(x, device)
    if not isinstance(weights, Tensor) or len(weights.shape) != 2 or weights.data is None:
        raise ValueError(f"{operation} requires initialized weights with shape [K,N]")


def evaluate_matvec(
    x: tuple[int, ...], weights: Tensor, device: ReferenceDevice
) -> KernelResult:
    """Compute x[K] * weights[K,N], retaining exact sums across K tiles."""
    _validate_product(x, weights, device, "matvec")
    if len(x) != weights.shape[0]:
        raise ValueError("matvec input length must match weights K dimension")
    return _evaluate_product(x, weights, device, rows=1, matrix=False)


def evaluate_matmul(
    x: tuple[int, ...], weights: Tensor, device: ReferenceDevice
) -> KernelResult:
    """Compute flat row-major A[M,K] @ W[K,N]; infer M from len(A)/K.

    Complete A and Y buffers stay resident. Each sequential row loads W again,
    producing row-indexed weight events. Only final sums must fit signed int64.
    """
    _validate_product(x, weights, device, "matmul")
    k_size, n_size = weights.shape
    if len(x) % k_size:
        raise ValueError("matmul flat input length must be a positive multiple of weights K")
    rows = len(x) // k_size
    if rows * n_size > 1_000_000 or rows * k_size * n_size > 2_000_000:
        raise ValueError("matmul exceeds the reference output or work limit")
    return _evaluate_product(x, weights, device, rows, matrix=True)


def evaluate_relu(x: tuple[int, ...], device: ReferenceDevice) -> KernelResult:
    """Apply ReLU to a flat rank-1/2 activation, retaining its complete input."""
    _validate_vector(x, device)
    _check_capacity(16 * len(x), device)
    return KernelResult(
        tuple(checked_integer(max(0, value)) for value in x),
        ({"kind": "vector_relu", "elements": len(x)},),
    )


def evaluate_add(
    x: tuple[int, ...], y: tuple[int, ...], device: ReferenceDevice
) -> KernelResult:
    """Elementwise addition with full buffers; aliased operands share storage."""
    _validate_vector(x, device)
    _validate_vector(y, device)
    if len(x) != len(y):
        raise ValueError("add inputs must have matching flat lengths")
    _check_capacity(8 * (len(x) * 2 + (0 if x is y else len(y))), device)
    return KernelResult(
        tuple(checked_integer(a + b) for a, b in zip(x, y)),
        ({"kind": "elementwise_add", "elements": len(x)},),
    )
