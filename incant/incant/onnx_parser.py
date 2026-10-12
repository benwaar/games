"""ONNX model parser — extract layers, weights, biases, activations.

Parses a float32 ONNX model (Linear → ReLU → Linear → ...) and returns
a structured dict suitable for quantisation and Z80 code generation.

Requires: onnx, onnxruntime, numpy (all optional — functions raise
ImportError with a clear message if unavailable).
"""

from __future__ import annotations

from typing import Any


def _require_deps() -> tuple[Any, Any, Any]:
    """Import onnx, onnxruntime, numpy — raise ImportError if missing."""
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort

        return onnx, ort, np
    except ImportError as e:
        raise ImportError(
            f"M10 requires onnx, onnxruntime, numpy: pip install onnx onnxruntime numpy. "
            f"Missing: {e}"
        ) from e


def parse_onnx(path: str) -> dict:
    """Parse an ONNX model and extract its layer graph.

    Supports: Linear (Gemm/MatMul+Add) layers and ReLU activations.
    Returns a dict:
    {
        "layers": [
            {"type": "linear", "in": 80, "out": 32,
             "weight": np.ndarray (out×in, float32),
             "bias":   np.ndarray (out, float32)},
            {"type": "relu"},
            ...
        ],
        "input_shape":  [int, ...],
        "output_shape": [int, ...],
    }
    """
    onnx_mod, _ort, np = _require_deps()

    model = onnx_mod.load(path)
    onnx_mod.checker.check_model(model)

    graph = model.graph

    # Build initialiser lookup (name → numpy array)
    inits: dict[str, Any] = {}
    for init in graph.initializer:
        arr = _tensor_to_numpy(init, np)
        inits[init.name] = arr

    # Walk nodes in order
    layers: list[dict] = []
    for node in graph.node:
        op = node.op_type

        if op == "Gemm":
            # inputs: [X, W, B?]
            w_name = node.input[1]
            b_name = node.input[2] if len(node.input) > 2 else None
            weight = inits[w_name]  # shape (out, in) for transB=1 (default)
            bias = inits[b_name] if b_name and b_name in inits else np.zeros(weight.shape[0], dtype=np.float32)
            # Handle transA/transB attributes
            trans_b = 1
            for attr in node.attribute:
                if attr.name == "transB":
                    trans_b = attr.i
            if trans_b == 0:
                weight = weight.T
            layers.append({
                "type": "linear",
                "in": int(weight.shape[1]),
                "out": int(weight.shape[0]),
                "weight": weight.astype(np.float32),
                "bias": bias.astype(np.float32),
            })

        elif op == "MatMul":
            # May be followed by Add for bias — merge below
            w_name = node.input[1]
            weight = inits.get(w_name)
            if weight is None:
                continue  # dynamic weights — skip
            layers.append({
                "type": "_matmul_pending",
                "in": int(weight.shape[0]),
                "out": int(weight.shape[1]),
                "weight": weight.astype(np.float32),
                "bias": None,
                "_output_name": node.output[0],
            })

        elif op == "Add":
            # If previous layer is _matmul_pending and this Add uses that output → merge
            if (
                layers
                and layers[-1]["type"] == "_matmul_pending"
                and node.input[0] == layers[-1]["_output_name"]
            ):
                b_name = node.input[1]
                bias = inits.get(b_name, None)
                prev = layers[-1]
                prev["type"] = "linear"
                prev["bias"] = (
                    bias.astype(np.float32)
                    if bias is not None
                    else __import__("numpy").zeros(prev["out"], dtype=__import__("numpy").float32)
                )
                del prev["_output_name"]
            # else: plain Add, ignore

        elif op == "Relu":
            layers.append({"type": "relu"})

        # Other ops (Flatten, Dropout, etc.) — silently skip

    # Finalise any dangling _matmul_pending (no bias Add followed)
    for layer in layers:
        if layer.get("type") == "_matmul_pending":
            import numpy as np2  # noqa: F811
            layer["type"] = "linear"
            layer["bias"] = np2.zeros(layer["out"], dtype=np2.float32)
            del layer["_output_name"]

    # Determine input / output shapes from graph I/O
    input_shape = _shape_from_value_info(graph.input[0]) if graph.input else []
    output_shape = _shape_from_value_info(graph.output[0]) if graph.output else []

    return {
        "layers": layers,
        "input_shape": input_shape,
        "output_shape": output_shape,
    }


def _tensor_to_numpy(tensor: Any, np: Any) -> Any:
    """Convert an ONNX TensorProto to a numpy array."""
    import onnx.numpy_helper as nph  # type: ignore[import-untyped]

    return nph.to_array(tensor)


def _shape_from_value_info(vi: Any) -> list[int]:
    """Extract shape dimensions from a ValueInfoProto, skipping dynamic dims."""
    shape = []
    try:
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                shape.append(int(dim.dim_value))
            else:
                shape.append(-1)  # dynamic
    except Exception:
        pass
    return shape


def architecture_summary(parsed: dict) -> str:
    """Return a human-readable architecture string, e.g. '4 → ReLU → 3 → 2'."""
    parts: list[str] = []
    for layer in parsed["layers"]:
        if layer["type"] == "linear":
            if not parts:
                parts.append(str(layer["in"]))
            parts.append(str(layer["out"]))
        elif layer["type"] == "relu":
            # Insert ReLU between the last linear and next
            if parts:
                parts[-1] = parts[-1] + " → ReLU"
    return " → ".join(parts)
