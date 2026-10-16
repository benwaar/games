"""incant CLI — cast sigils into binaries."""

import argparse
import re
import sys
from pathlib import Path

from .gen import cast_multi, cast_sigil
from .harness import WasmHarness, Z80Harness
from .manifest import parse_manifest
from .rag import VectorStore
from .sigil import parse_sigil
from .smelt import (
    decompose_spec,
    load_catalogue,
    parse_spec,
    write_sigils_and_manifest,
)
from .tap import disassemble, extract_code_blocks, read_tap
from .targets import wat as wat_target


def cmd_infer(args):
    """ONNX → inference engine (Z80 INT8 assembly or WASM f32)."""
    try:
        import numpy as np
    except ImportError:
        print("Error: numpy required — pip install numpy", file=sys.stderr)
        sys.exit(1)

    try:
        from .onnx_parser import architecture_summary, parse_onnx
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: {model_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {model_path} ...")
    parsed = parse_onnx(str(model_path))
    print(f"  Architecture: {architecture_summary(parsed)}")

    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = model_path.stem

    if args.target == "wasm":
        _cmd_infer_wasm(args, parsed, model_path, output_dir, stem, np)
    else:
        _cmd_infer_z80(args, parsed, model_path, output_dir, stem, np)


def _cmd_infer_wasm(args, parsed, model_path, output_dir, stem, np):
    try:
        from .wasm_infer import generate_wat_inference
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Generating WAT (f32) ...")
    wat_source = generate_wat_inference(parsed)

    ok, wasm_path, err = wat_target.assemble(wat_source)
    if not ok:
        print(f"  FAILED to assemble: {err}", file=sys.stderr)
        sys.exit(1)

    import shutil

    wat_out = output_dir / f"{stem}.wat"
    wasm_out = output_dir / f"{stem}.wasm"
    wat_out.write_text(wat_source)
    shutil.copy2(wasm_path, wasm_out)
    print(f"  WAT:  {wat_out}")
    print(f"  WASM: {wasm_out} ({wasm_out.stat().st_size} bytes)")

    if getattr(args, "accuracy", False):
        import struct
        import subprocess
        from pathlib import Path as _P

        runner = _P(__file__).parent.parent / "tests" / "wasm_infer_runner.js"
        if not runner.exists():
            print("  Skipped WASM accuracy (wasm_infer_runner.js not found)")
            return
        try:
            import onnxruntime as ort

            in_dim = parsed["input_shape"][-1] if parsed["input_shape"] else None
            if not (in_dim and in_dim > 0):
                print("  Skipped WASM accuracy (unknown input dim)")
                return
            sess = ort.InferenceSession(str(model_path))
            in_name = sess.get_inputs()[0].name
            rng = np.random.default_rng(0)
            agree, n = 0, 100
            for _ in range(n):
                x = rng.standard_normal((1, in_dim)).astype(np.float32)
                ref_am = int(np.argmax(sess.run(None, {in_name: x})[0].flatten()))
                hex_args = [struct.pack("<f", float(v)).hex() for v in x.flatten()]
                r = subprocess.run(
                    ["node", str(runner), str(wasm_out)] + hex_args,
                    capture_output=True, text=True,
                )
                if r.returncode == 0:
                    agree += int(int(r.stdout.strip()) == ref_am)
            print(f"  WASM argmax agreement: {agree}/{n} = {agree/n:.0%}")
        except ImportError as e:
            print(f"  Skipped WASM accuracy (missing dep): {e}")
        except Exception as e:
            print(f"  WASM accuracy test failed: {e}")


def _cmd_infer_z80(args, parsed, model_path, output_dir, stem, np):
    try:
        from .quantise import compare_accuracy, quantise_model
        from .z80_infer import generate_z80_inference, run_z80_inference
        from .targets.z80 import assemble as z80_assemble
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Quantising to INT8 ...")
    quantised = quantise_model(parsed)
    linear_layers = [l for l in quantised["layers"] if l["type"] == "linear"]
    for i, layer in enumerate(linear_layers):
        print(f"  FC{i+1}: {layer['in']}x{layer['out']}, scale_w={layer['scale_w']:.5f}")

    print("Generating Z80 assembly ...")
    asm_source = generate_z80_inference(quantised)

    ok, binary, err = z80_assemble(asm_source)
    if not ok:
        print(f"  FAILED to assemble: {err}", file=sys.stderr)
        sys.exit(1)
    print(f"  Assembled: {len(binary)} bytes")

    asm_out = output_dir / f"{stem}_z80_infer.asm"
    bin_out = output_dir / f"{stem}_z80_infer.bin"
    asm_out.write_text(asm_source)
    bin_out.write_bytes(binary)
    print(f"  ASM: {asm_out}")
    print(f"  BIN: {bin_out}")

    if getattr(args, "accuracy", False):
        print("Running accuracy test (100 samples) ...")
        try:
            result = compare_accuracy(parsed, quantised, str(model_path), n_samples=100)
            print(f"  Argmax agreement (INT8 ref vs float32): {result['argmax_agreement']:.0%}")
            print(f"  Mean L1 error: {result['mean_l1_error']:.4f}")
        except ImportError as e:
            print(f"  Skipped (missing dep): {e}")

        print("Running Z80 accuracy test (20 samples) ...")
        try:
            import onnxruntime as ort

            in_dim = parsed["input_shape"][-1] if parsed["input_shape"] else None
            if in_dim and in_dim > 0:
                sess = ort.InferenceSession(str(model_path))
                in_name = sess.get_inputs()[0].name
                rng = np.random.default_rng(0)
                agree = 0
                n = 20
                for _ in range(n):
                    x = rng.uniform(-1.0, 1.0, size=(1, in_dim)).astype(np.float32)
                    ref_am = int(np.argmax(sess.run(None, {in_name: x})[0].flatten()))
                    x_int8 = np.clip(np.round(x.flatten() * 127.0), -128, 127).astype(np.int8)
                    x_bytes = bytes([b & 0xFF for b in x_int8.tolist()])
                    z80_am = run_z80_inference(quantised, x_bytes, ticks=10_000_000)
                    agree += int(z80_am == ref_am)
                print(f"  Z80 argmax agreement: {agree}/{n} = {agree/n:.0%}")
        except ImportError as e:
            print(f"  Skipped (missing dep): {e}")
        except Exception as e:
            print(f"  Z80 accuracy test failed: {e}")

PROJECT_ROOT = Path(__file__).parent.parent
STORE_PATH = PROJECT_ROOT / "data" / "embeddings.jsonl"


def cmd_run(args):
    path = Path(args.binary)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    if path.suffix == ".wasm":
        harness = WasmHarness(path)
    elif path.suffix == ".bin":
        harness = Z80Harness(path)
    else:
        print(f"Error: unsupported binary type '{path.suffix}' — use .wasm or .bin", file=sys.stderr)
        sys.exit(1)

    try:
        output = harness.run(args.input)
    except (TimeoutError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(output, end="")


def cmd_cast(args):
    store = VectorStore(STORE_PATH)
    sigil = parse_sigil(Path(args.sigil))
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "output"

    print(f"Casting sigil: {sigil.name} (target: {sigil.target})")

    success, message = cast_sigil(
        sigil, store, output_dir, verbose=args.verbose,
        write_tap=getattr(args, "tap", False),
    )

    if success:
        print(f"  {message}")
        print(f"  Output: {output_dir / sigil.name}.*")
    else:
        print(f"  FAILED: {message}", file=sys.stderr)
        sys.exit(1)


def cmd_multi(args):
    store = VectorStore(STORE_PATH)
    manifest = parse_manifest(Path(args.manifest))
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "output"

    print(f"Multi-cast: {manifest.name} ({len(manifest.sigils)} sigils, target: {manifest.target})")

    success, message = cast_multi(
        manifest, store, output_dir, verbose=args.verbose
    )

    if success:
        print(f"  {message}")
        print(f"  Output: {output_dir / manifest.name}.*")
    else:
        print(f"  FAILED: {message}", file=sys.stderr)
        sys.exit(1)


def cmd_smelt(args):
    store = VectorStore(STORE_PATH)
    spec = parse_spec(Path(args.spec))
    output_dir = Path(args.output) if args.output else PROJECT_ROOT / "output"
    libs_dir = PROJECT_ROOT / "sigils" / "libs"
    smelt_dir = output_dir / "smelt"

    print(f"Smelting spec: {spec.title} ({len(spec.scenarios)} scenarios, target: {spec.target})")

    # Load library catalogue
    catalogue = load_catalogue(libs_dir)
    lib_count = sum(len(v) for v in catalogue.values())
    print(f"  Library: {lib_count} functions available")

    # LLM decomposition
    print("  Decomposing spec into sigils...")
    sigil_dicts, manifest_dict = decompose_spec(
        spec, catalogue, verbose=args.verbose,
    )
    print(f"  Generated {len(sigil_dicts)} sigils")

    # Write to disk
    manifest_path = write_sigils_and_manifest(
        sigil_dicts, manifest_dict, smelt_dir, libs_dir,
    )
    print(f"  Manifest: {manifest_path}")

    # Run the existing pipeline
    manifest = parse_manifest(manifest_path)
    success, message = cast_multi(manifest, store, output_dir, verbose=args.verbose)

    if not success:
        print(f"  BUILD FAILED: {message}", file=sys.stderr)
        sys.exit(1)

    print(f"  {message}")

    # For WASI programs, run scenario tests
    wasi_sigils = [s for s in sigil_dicts if s.get("wasi")]
    if wasi_sigils:
        wasm_name = manifest_dict.get("name", "program")
        wasm_path = output_dir / f"{wasm_name}.wasm"
        if not wasm_path.exists():
            # Try assembling the combined WAT
            wat_path = output_dir / f"{wasm_name}.wat"
            if wat_path.exists():
                ok, wasm_path, err = wat_target.assemble(wat_path.read_text())
                if not ok:
                    print(f"  WASI assembly failed: {err}", file=sys.stderr)
                    sys.exit(1)

        if wasm_path and wasm_path.exists():
            print("  Running WASI scenario tests...")
            all_passed = True
            for scenario in spec.scenarios:
                stdin_parts = []
                for w in scenario.when:
                    match = re.search(r'"([^"]*)"', w)
                    if match:
                        stdin_parts.append(match.group(1))
                stdin_input = "\n".join(stdin_parts) if stdin_parts else ""

                expected_parts = []
                for t in scenario.then:
                    match = re.search(r'"([^"]*)"', t)
                    if match:
                        expected_parts.append(match.group(1))
                expected_stdout = "".join(expected_parts)
                # Unescape \n in expected output
                expected_stdout = expected_stdout.replace("\\n", "\n")

                passed, msg = wat_target.run_wasi_test(
                    wasm_path, stdin_input, expected_stdout,
                )
                status = "PASS" if passed else "FAIL"
                print(f"    {scenario.name}: {status}")
                if not passed:
                    print(f"      {msg}")
                    all_passed = False

            if not all_passed:
                print("  Some scenario tests failed", file=sys.stderr)
                sys.exit(1)
            print(f"  All {len(spec.scenarios)} scenarios passed")

    print(f"  Output: {output_dir / manifest_dict.get('name', 'program')}.*")


def cmd_disasm(args):
    tap_path = Path(args.tap)
    if not tap_path.exists():
        print(f"Error: {tap_path} not found", file=sys.stderr)
        sys.exit(1)

    embed = getattr(args, "embed", False)
    store = VectorStore(STORE_PATH) if embed else None

    code_blocks = extract_code_blocks(str(tap_path))
    if not code_blocks:
        # No typed header found — try raw blocks as data
        blocks = read_tap(str(tap_path))
        data_blocks = [b for b in blocks if b["flag"] == 0xFF]
        if not data_blocks:
            print("No code or data blocks found in TAP file.", file=sys.stderr)
            sys.exit(1)
        code_blocks = [
            {"name": f"block_{i}", "start_addr": 0x8000, "data": b["payload"]}
            for i, b in enumerate(data_blocks)
        ]

    all_asm_lines = []
    for block in code_blocks:
        name = block["name"]
        addr = block["start_addr"]
        asm = disassemble(block["data"], addr)
        header = f"; --- {name} (load addr: 0x{addr:04X}, {len(block['data'])} bytes) ---"
        block_text = f"{header}\n{asm}"
        print(block_text)
        print()
        all_asm_lines.append(block_text)

    if embed and store is not None:
        _embed_disasm(store, str(tap_path), all_asm_lines)
        print(f"Embedded {len(all_asm_lines)} block(s) into z80 collection.")


def _embed_disasm(store: "VectorStore", tap_path: str, asm_blocks: list[str]) -> None:
    """Chunk disassembled blocks and embed into the z80 RAG collection."""
    from .rag import Chunk, embed_text

    source_name = Path(tap_path).stem
    for i, block_text in enumerate(asm_blocks):
        lines = block_text.splitlines()
        heading = lines[0].lstrip("; ").strip() if lines else f"block_{i}"
        embedding = embed_text(f"{heading}\n{block_text}")
        chunk = Chunk(
            source=f"{source_name}.tap",
            collection="z80",
            heading=heading,
            text=block_text,
            embedding=embedding,
        )
        store.chunks.append(chunk)
    store._save()


def cmd_rag(args):
    store = VectorStore(STORE_PATH)

    if args.rag_action == "embed":
        dir_path = Path(args.path)
        collection = args.collection
        append = getattr(args, "append", False)
        print(f"Embedding {dir_path} into collection '{collection}'{'(append)' if append else ''}...")
        store.embed_directory(dir_path, collection, replace=not append)
        count = sum(1 for c in store.chunks if c.collection == collection)
        print(f"  {count} chunks in collection")

    elif args.rag_action == "query":
        query = args.query_text
        collection = args.collection
        print(f"Query: '{query}' (collection: {collection or 'all'})")
        results = store.query(query, collection=collection, top_k=args.top_k)
        for chunk, score in results:
            print(f"\n  [{score:.3f}] {chunk.heading} ({chunk.source})")
            preview = chunk.text[:200].replace("\n", " ")
            print(f"  {preview}...")


def main():
    parser = argparse.ArgumentParser(
        prog="incant",
        description="Spec-driven code generation for WAT and Z80 assembly",
    )
    subparsers = parser.add_subparsers(dest="command")

    # cast
    cast_parser = subparsers.add_parser("cast", help="Generate code from a sigil")
    cast_parser.add_argument("sigil", help="Path to sigil YAML file")
    cast_parser.add_argument("-o", "--output", help="Output directory")
    cast_parser.add_argument("-v", "--verbose", action="store_true")
    cast_parser.add_argument("--tap", action="store_true", help="Also write .tap for Z80 targets")
    cast_parser.set_defaults(func=cmd_cast)

    # multi
    multi_parser = subparsers.add_parser("multi", help="Multi-sigil build from manifest")
    multi_parser.add_argument("manifest", help="Path to manifest YAML file")
    multi_parser.add_argument("-o", "--output", help="Output directory")
    multi_parser.add_argument("-v", "--verbose", action="store_true")
    multi_parser.set_defaults(func=cmd_multi)

    # smelt
    smelt_parser = subparsers.add_parser("smelt", help="BDD spec → sigils → binary")
    smelt_parser.add_argument("spec", help="Path to .spec.md file")
    smelt_parser.add_argument("-o", "--output", help="Output directory")
    smelt_parser.add_argument("-v", "--verbose", action="store_true")
    smelt_parser.set_defaults(func=cmd_smelt)

    # disasm
    disasm_parser = subparsers.add_parser("disasm", help="Disassemble Z80 code from a TAP file")
    disasm_parser.add_argument("tap", help="Path to .tap file")
    disasm_parser.add_argument(
        "--embed",
        action="store_true",
        help="Embed disassembled output into the z80 RAG collection",
    )
    disasm_parser.set_defaults(func=cmd_disasm)

    # rag
    rag_parser = subparsers.add_parser("rag", help="Manage RAG knowledge base")
    rag_sub = rag_parser.add_subparsers(dest="rag_action")

    embed_parser = rag_sub.add_parser("embed", help="Embed a directory")
    embed_parser.add_argument("path", help="Directory of markdown files")
    embed_parser.add_argument("collection", help="Collection name")
    embed_parser.add_argument("--append", action="store_true", help="Append to existing collection instead of replacing")

    query_parser = rag_sub.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query_text", help="Query string")
    query_parser.add_argument("-c", "--collection", help="Filter by collection")
    query_parser.add_argument("-k", "--top-k", type=int, default=5)

    rag_parser.set_defaults(func=cmd_rag)

    # run
    run_parser = subparsers.add_parser("run", help="Run a compiled binary with string input")
    run_parser.add_argument("binary", help="Path to .wasm or .bin file")
    run_parser.add_argument("input", help="String to pass as input")
    run_parser.set_defaults(func=cmd_run)

    # infer
    infer_parser = subparsers.add_parser("infer", help="ONNX → inference engine (z80 or wasm)")
    infer_parser.add_argument("model", help="Path to .onnx model file")
    infer_parser.add_argument(
        "--target", choices=["z80", "wasm"], default="z80", help="Target architecture (default: z80)"
    )
    infer_parser.add_argument(
        "--accuracy", action="store_true", help="Run accuracy test against float32 reference"
    )
    infer_parser.add_argument("-o", "--output", help="Output directory")
    infer_parser.set_defaults(func=cmd_infer)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
