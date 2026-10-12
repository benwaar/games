"""Unified test harness for WASM and Z80 compiled binaries.

Both harnesses accept a path to a compiled binary and expose a single
.run(input_str) -> str interface — same call, different backends.
"""

import subprocess
from pathlib import Path

import z80 as z80lib

INPUT_BUF = 0x8000
OUTPUT_BUF = 0x9000
_MAX_OUTPUT = 256
_DEFAULT_TICKS = 100_000


class Z80Harness:
    """Run a compiled Z80 .bin against memory-mapped string I/O."""

    def __init__(self, binary_path: str | Path, ticks: int = _DEFAULT_TICKS) -> None:
        self._data = Path(binary_path).read_bytes()
        self._ticks = ticks

    def run(self, input_str: str) -> str:
        machine = z80lib.Z80Machine()
        machine.set_memory_block(0, self._data)
        machine.ticks_to_stop = self._ticks
        machine.set_memory_block(INPUT_BUF, input_str.encode("ascii") + b"\x00")
        machine.run()
        if not machine.halted:
            raise TimeoutError("Z80 program did not halt within tick limit")
        raw = []
        for i in range(_MAX_OUTPUT):
            b = machine.memory[OUTPUT_BUF + i]
            if b == 0:
                break
            raw.append(b)
        return bytes(raw).decode("ascii", errors="replace")


class WasmHarness:
    """Run a compiled WASI .wasm against stdin/stdout."""

    def __init__(self, binary_path: str | Path, timeout: int = 30) -> None:
        self._path = Path(binary_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Binary not found: {self._path}")
        self._timeout = timeout

    def run(self, input_str: str) -> str:
        try:
            result = subprocess.run(
                ["wasm-interp", "--wasi", str(self._path)],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired as e:
            raise TimeoutError("wasm-interp --wasi timed out") from e
        if result.returncode != 0:
            raise RuntimeError(f"wasm-interp failed: {result.stderr.strip()}")
        return result.stdout
