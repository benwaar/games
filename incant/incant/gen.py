"""Code generation orchestrator — sigil + RAG context → LLM → code."""

from pathlib import Path

import ollama

from .manifest import Manifest, topological_sort
from .rag import VectorStore
from .sigil import Sigil
from .stitch import stitch_wat, stitch_z80
from .targets import z80 as z80_target
from .targets import wat as wat_target

GEN_MODEL = "qwen3-coder:latest"
MAX_RETRIES = 3

TARGETS = {
    "z80": z80_target,
    "wat": wat_target,
}


def get_rag_context(store: VectorStore, sigil: Sigil, top_k: int = 5) -> str:
    """Retrieve relevant knowledge for the sigil's target."""
    query = f"{sigil.description} {sigil.target}"
    results = store.query(query, collection=sigil.target, top_k=top_k)

    if not results:
        return ""

    chunks = []
    for chunk, score in results:
        chunks.append(f"### {chunk.heading}\n{chunk.text}")

    return "## Reference documentation\n\n" + "\n\n".join(chunks)


def call_llm(system: str, prompt: str) -> str:
    response = ollama.chat(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


def cast_sigil(
    sigil: Sigil,
    store: VectorStore,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Generate code for a sigil. Returns (success, message)."""
    target = TARGETS.get(sigil.target)
    if not target:
        return False, f"Unknown target: {sigil.target}"

    rag_context = get_rag_context(store, sigil)
    prompt = target.build_prompt(sigil, rag_context)

    if verbose:
        print(f"  RAG context: {len(rag_context)} chars")
        print(f"  Prompt: {len(prompt)} chars")

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"  Attempt {attempt}/{MAX_RETRIES}...")

        if last_error:
            retry_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt failed with error:\n{last_error}\n\n"
                f"Fix the error and output ONLY the corrected code."
            )
        else:
            retry_prompt = prompt

        raw_output = call_llm(target.SYSTEM_PROMPT, retry_prompt)
        code = target.extract_code(raw_output)

        if verbose:
            print(f"  Generated {len(code)} chars")

        # gate: assemble
        success, artifact, error = target.assemble(code)
        if not success:
            last_error = error
            if verbose:
                print(f"  Assembly failed: {error}")
            continue

        # gate: run tests
        all_passed = True
        for i, test in enumerate(sigil.tests):
            passed, msg = target.run_test(artifact, test, sigil)
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  Test {i + 1}: {status} — {msg}")
            if not passed:
                all_passed = False
                last_error = f"Test {i + 1} failed: {msg}"
                break

        if all_passed:
            # save output
            output_dir.mkdir(parents=True, exist_ok=True)
            ext = ".asm" if sigil.target == "z80" else ".wat"
            code_path = output_dir / f"{sigil.name}{ext}"
            code_path.write_text(code)

            if sigil.target == "z80":
                bin_path = output_dir / f"{sigil.name}.bin"
                bin_path.write_bytes(artifact)
            # for WAT, the .wasm is already at artifact path

            return True, f"All {len(sigil.tests)} tests passed"

    return False, f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"


def _generate_code(
    sigil: Sigil,
    store: VectorStore,
    dep_context: str = "",
    verbose: bool = False,
) -> tuple[bool, str, str | None]:
    """Generate code for one sigil. Returns (success, code, error)."""
    target = TARGETS.get(sigil.target)
    if not target:
        return False, "", f"Unknown target: {sigil.target}"

    rag_context = get_rag_context(store, sigil)
    if dep_context:
        rag_context = f"{rag_context}\n\n## Available functions\n\n{dep_context}"

    prompt = target.build_prompt(sigil, rag_context)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if verbose:
            print(f"    Attempt {attempt}/{MAX_RETRIES}...")

        if last_error:
            retry_prompt = (
                f"{prompt}\n\n"
                f"Previous attempt failed with error:\n{last_error}\n\n"
                f"Fix the error and output ONLY the corrected code."
            )
        else:
            retry_prompt = prompt

        raw_output = call_llm(target.SYSTEM_PROMPT, retry_prompt)
        code = target.extract_code(raw_output)

        success, artifact, error = target.assemble(code)
        if not success:
            last_error = error
            if verbose:
                print(f"    Assembly failed: {error}")
            continue

        all_passed = True
        for i, test in enumerate(sigil.tests):
            passed, msg = target.run_test(artifact, test, sigil)
            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"    Test {i + 1}: {status} — {msg}")
            if not passed:
                all_passed = False
                last_error = f"Test {i + 1} failed: {msg}"
                break

        if all_passed:
            return True, code, None

    return False, "", f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"


def cast_multi(
    manifest: Manifest,
    store: VectorStore,
    output_dir: Path,
    verbose: bool = False,
) -> tuple[bool, str]:
    """Generate code for all sigils in a manifest, stitch into one program."""
    sorted_sigils = topological_sort(manifest.sigils)
    target = TARGETS.get(manifest.target)
    if not target:
        return False, f"Unknown target: {manifest.target}"

    generated_code: dict[str, str] = {}

    for sigil in sorted_sigils:
        if verbose:
            print(f"  Generating: {sigil.name}")

        dep_context = ""
        if sigil.dependencies:
            dep_parts = []
            for dep_name in sigil.dependencies:
                if dep_name in generated_code:
                    dep_parts.append(f"```\n{generated_code[dep_name]}\n```")
            if dep_parts:
                dep_context = "\n\n".join(dep_parts)

        success, code, error = _generate_code(
            sigil, store, dep_context=dep_context, verbose=verbose,
        )

        if not success:
            return False, f"Sigil '{sigil.name}' failed: {error}"

        generated_code[sigil.name] = code
        if verbose:
            print(f"  {sigil.name}: OK ({len(code)} chars)")

    # stitch
    codes = [generated_code[s.name] for s in sorted_sigils]
    names = [s.name for s in sorted_sigils]

    if manifest.target == "wat":
        combined = stitch_wat(codes)
    elif manifest.target == "z80":
        combined = stitch_z80(codes, names)
    else:
        return False, f"Unknown target: {manifest.target}"

    # assemble combined
    success, artifact, error = target.assemble(combined)
    if not success:
        return False, f"Combined assembly failed: {error}"

    # verify combined binary
    # WAT: each function is independently callable, so re-test all
    # Z80: programs share registers/state, so individual tests already
    #      passed during generation — just verify assembly succeeds
    if manifest.target == "wat":
        for sigil in sorted_sigils:
            for i, test in enumerate(sigil.tests):
                passed, msg = target.run_test(artifact, test, sigil)
                if not passed:
                    return False, f"Combined test failed for '{sigil.name}' test {i + 1}: {msg}"

    # save output
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = ".asm" if manifest.target == "z80" else ".wat"
    code_path = output_dir / f"{manifest.name}{ext}"
    code_path.write_text(combined)

    if manifest.target == "z80":
        bin_path = output_dir / f"{manifest.name}.bin"
        bin_path.write_bytes(artifact)

    return True, f"All sigils generated and stitched ({len(sorted_sigils)} functions)"
