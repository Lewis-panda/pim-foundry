"""Portable, strict JSON configuration for logical resources and cost models."""

from dataclasses import asdict
import json
from pathlib import Path

from .core import Platform, ReferenceDevice


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("JSON input exceeds the 32 MiB limit")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject)


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def exact_fields(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{label} must have exactly these fields: {', '.join(sorted(keys))}")


def platform_from_dict(data):
    exact_fields(data, ("devices", "links"), "platform")
    if type(data["devices"]) is not list or not 1 <= len(data["devices"]) <= 16:
        raise ValueError("platform needs 1 to 16 devices")
    if type(data["links"]) is not list or len(data["links"]) > 272:
        raise ValueError("platform links must be an array with at most 272 links")
    devices = []
    for item in data["devices"]:
        exact_fields(item, ("name", "kind", "capacity_bytes", "tile_k", "tile_n", "units"), "device")
        devices.append(ReferenceDevice(**item))
    if any(type(link) is not list for link in data["links"]):
        raise ValueError("platform links must contain arrays")
    platform = Platform(tuple(devices), tuple(tuple(link) for link in data["links"]))
    platform.validate()
    return platform


def validate_config(data):
    from .simulation import validate_model
    exact_fields(data, ("schema_version", "name", "platform", "model"), "configuration")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported configuration schema")
    if type(data["name"]) is not str or not data["name"].strip():
        raise ValueError("configuration name must be nonempty")
    platform = platform_from_dict(data["platform"])
    validate_model(platform, data["model"])
    return platform


def example_config(kinds=("hbm", "gpu")):
    from .simulation import example_model
    allowed = ("sram", "hbm", "gpu", "npu")
    if (type(kinds) is not tuple or not kinds or any(type(k) is not str or k not in allowed for k in kinds)
            or len(set(kinds)) != len(kinds)):
        raise ValueError("devices must be unique sram, hbm, gpu or npu options")
    # These dimensions are educational configuration values, not product specs.
    dimensions = {"sram": (256 * 1024, 16, 16, 4), "hbm": (8 * 1024**2, 32, 16, 8),
                  "gpu": (32 * 1024**2, 32, 32, 16), "npu": (4 * 1024**2, 16, 32, 8)}
    selected = tuple(k for k in allowed if k in kinds)
    endpoints = ("host", *selected)
    platform = Platform(tuple(ReferenceDevice(k, k, *dimensions[k]) for k in selected),
                        tuple((a, b) for a in endpoints for b in endpoints if a != b))
    return {"schema_version": 1, "name": "synthetic-" + "-".join(selected),
            "platform": json_value(asdict(platform)), "model": example_model(platform)}
