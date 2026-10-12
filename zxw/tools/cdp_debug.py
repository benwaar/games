#!/usr/bin/env python3
"""CDP debug client for zxw — inspect and patch the running Spectrum emulator.

Connects to Chrome via the Chrome DevTools Protocol WebSocket, then evaluates
JavaScript expressions against window.zxw.* to read/write emulator state.

Requirements:
    pip install websockets

Usage:
    # Start Chrome with remote debugging enabled:
    #   Google Chrome --remote-debugging-port=9222
    #   chromium --remote-debugging-port=9222
    #
    # Then in another terminal:
    python3 tools/cdp_debug.py            # default port 9222
    python3 tools/cdp_debug.py 9333       # custom port

Commands (at the zxw> prompt):
    regs              — print all Z80 registers
    mem ADDR [LEN]    — hex dump LEN bytes from ADDR (default 16)
    poke ADDR B1 ...  — write one or more bytes to ADDR
    bp ADDR           — set breakpoint at ADDR
    bp clear [ADDR]   — clear one breakpoint (or all if ADDR omitted)
    pause             — pause emulation
    resume            — resume emulation
    step [N]          — single-step N instructions (default 1)
    screen [FILE]     — dump screen memory (0x4000, 6912 bytes); save to FILE if given
    watch ADDR [LEN]  — repeatedly read memory at ADDR every 0.5 s (Ctrl-C to stop)
    hits              — print accumulated watchpoint hits, then clear them
    watch_add ADDR    — register a write watchpoint at ADDR
    watch_rm  ADDR    — remove a write watchpoint at ADDR
    help              — show this message
    quit / exit       — disconnect and quit
"""

import asyncio
import json
import sys
import urllib.request

try:
    import websockets
except ImportError:
    print("ERROR: 'websockets' library not found. Run: pip install websockets")
    sys.exit(1)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _addr(tok: str) -> int:
    """Parse hex (0xNNNN) or decimal address."""
    tok = tok.strip()
    if tok.startswith("0x") or tok.startswith("0X"):
        return int(tok, 16)
    return int(tok, 0)


def _hex_dump(data: list[int], base_addr: int) -> str:
    """Format bytes as an xxd-style hex dump."""
    lines = []
    for row in range(0, len(data), 16):
        chunk = data[row: row + 16]
        addr  = base_addr + row
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{addr:04x}  {hex_part:<47}  {asc_part}")
    return "\n".join(lines)


def _fmt_regs(r: dict) -> str:
    """Pretty-print the Z80 register object returned by window.zxw.regs()."""
    def h4(v): return f"{int(v):04x}"
    def h2(v): return f"{int(v):02x}"
    return (
        f"PC={h4(r['pc'])}  SP={h4(r['sp'])}  "
        f"AF={h4(r['af'])}  BC={h4(r['bc'])}  "
        f"DE={h4(r['de'])}  HL={h4(r['hl'])}\n"
        f"IX={h4(r['ix'])}  IY={h4(r['iy'])}  "
        f"AF'={h4(r['af2'])} BC'={h4(r['bc2'])} "
        f"DE'={h4(r['de2'])} HL'={h4(r['hl2'])}\n"
        f"I={h2(r['i'])}  R={h2(r['r'])}  "
        f"IFF1={int(r['iff1'])}  IFF2={int(r['iff2'])}  "
        f"IM={int(r['im'])}  HALT={int(r['halted'])}"
    )


# ── CDP communication ─────────────────────────────────────────────────────────

async def _get_ws_url(port: int) -> str:
    """Ask the CDP /json endpoint for the debugger WebSocket URL of the first page."""
    url = f"http://localhost:{port}/json"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            targets = json.loads(resp.read())
    except Exception as exc:
        print(f"ERROR: Cannot reach Chrome at {url}")
        print(f"  Make sure Chrome is running with --remote-debugging-port={port}")
        print(f"  Detail: {exc}")
        sys.exit(1)

    pages = [t for t in targets if t.get("type") == "page"]
    if not pages:
        print("ERROR: No page targets found via CDP. Open http://localhost:8080 in Chrome first.")
        sys.exit(1)

    ws_url = pages[0].get("webSocketDebuggerUrl", "")
    if not ws_url:
        print("ERROR: No webSocketDebuggerUrl in first page target.")
        sys.exit(1)

    return ws_url


class CdpClient:
    """Thin wrapper around the CDP Runtime.evaluate call."""

    def __init__(self, ws):
        self._ws = ws
        self._id = 0

    async def evaluate(self, expression: str, return_by_value: bool = True):
        """Send a Runtime.evaluate command and return the result."""
        self._id += 1
        msg = {
            "id": self._id,
            "method": "Runtime.evaluate",
            "params": {
                "expression": expression,
                "returnByValue": return_by_value,
                "awaitPromise": False,
            },
        }
        await self._ws.send(json.dumps(msg))
        # Drain messages until we see our reply.
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=10)
            data = json.loads(raw)
            if data.get("id") == self._id:
                result = data.get("result", {})
                if "exceptionDetails" in result:
                    raise RuntimeError(result["exceptionDetails"].get("text", "JS error"))
                rv = result.get("result", {})
                return rv.get("value")


# ── Command handlers ──────────────────────────────────────────────────────────

async def cmd_regs(cdp: CdpClient, _args):
    r = await cdp.evaluate("JSON.parse(JSON.stringify(window.zxw.regs()))")
    if r is None:
        print("ERROR: window.zxw not available — is the emulator page open?")
        return
    print(_fmt_regs(r))


async def cmd_mem(cdp: CdpClient, args):
    if not args:
        print("Usage: mem ADDR [LEN]")
        return
    addr = _addr(args[0])
    length = _addr(args[1]) if len(args) > 1 else 16
    expr = f"Array.from(window.zxw.mem({addr}, {length}))"
    data = await cdp.evaluate(expr)
    if data is None:
        print("ERROR: no data returned")
        return
    print(_hex_dump(data, addr))


async def cmd_poke(cdp: CdpClient, args):
    if len(args) < 2:
        print("Usage: poke ADDR B1 [B2 ...]")
        return
    addr  = _addr(args[0])
    bytes_str = ", ".join(str(_addr(b)) for b in args[1:])
    await cdp.evaluate(f"window.zxw.poke({addr}, {bytes_str})", return_by_value=False)
    print(f"Wrote {len(args) - 1} byte(s) to 0x{addr:04x}")


async def cmd_bp(cdp: CdpClient, args):
    if not args:
        print("Usage: bp ADDR  |  bp clear [ADDR]")
        return
    if args[0].lower() == "clear":
        if len(args) > 1:
            addr = _addr(args[1])
            await cdp.evaluate(f"window.zxw.clearBp({addr})", return_by_value=False)
            print(f"Cleared breakpoint at 0x{addr:04x}")
        else:
            await cdp.evaluate("window.zxw.clearAllBp()", return_by_value=False)
            print("Cleared all breakpoints")
    else:
        addr = _addr(args[0])
        await cdp.evaluate(f"window.zxw.bp({addr})", return_by_value=False)
        print(f"Breakpoint set at 0x{addr:04x}")


async def cmd_pause(cdp: CdpClient, _args):
    await cdp.evaluate("window.zxw.pause()", return_by_value=False)
    print("Emulation paused")


async def cmd_resume(cdp: CdpClient, _args):
    await cdp.evaluate("window.zxw.resume()", return_by_value=False)
    print("Emulation resumed")


async def cmd_step(cdp: CdpClient, args):
    n = int(args[0]) if args else 1
    for i in range(n):
        await cdp.evaluate("window.zxw.step()", return_by_value=False)
    # Show registers after stepping
    await cmd_regs(cdp, [])


async def cmd_screen(cdp: CdpClient, args):
    expr = "Array.from(window.zxw.screen())"
    data = await cdp.evaluate(expr)
    if data is None:
        print("ERROR: no data")
        return
    if args:
        path = args[0]
        with open(path, "wb") as fh:
            fh.write(bytes(data))
        print(f"Saved 6912 bytes of screen memory to {path}")
    else:
        # Print first 64 bytes as a quick sanity check
        print(f"Screen memory (first 64 of 6912 bytes):")
        print(_hex_dump(data[:64], 0x4000))


async def cmd_watch(cdp: CdpClient, args):
    if not args:
        print("Usage: watch ADDR [LEN]")
        return
    addr   = _addr(args[0])
    length = _addr(args[1]) if len(args) > 1 else 16
    expr   = f"Array.from(window.zxw.mem({addr}, {length}))"
    print(f"Watching 0x{addr:04x} ({length} bytes) — Ctrl-C to stop")
    try:
        while True:
            data = await cdp.evaluate(expr)
            if data is None:
                print("  (no data)")
            else:
                print(_hex_dump(data, addr))
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopped watching")


async def cmd_hits(cdp: CdpClient, _args):
    data = await cdp.evaluate("Array.from(window.zxw.hits())")
    if not data:
        print("No watchpoint hits")
        return
    triples = [(data[i], data[i+1], data[i+2]) for i in range(0, len(data), 3)]
    for addr, old, new in triples:
        print(f"  0x{addr:04x}: 0x{old:02x} → 0x{new:02x}")


async def cmd_watch_add(cdp: CdpClient, args):
    if not args:
        print("Usage: watch_add ADDR")
        return
    addr = _addr(args[0])
    await cdp.evaluate(f"window.zxw.watch({addr})", return_by_value=False)
    print(f"Watchpoint added at 0x{addr:04x}")


async def cmd_watch_rm(cdp: CdpClient, args):
    if not args:
        print("Usage: watch_rm ADDR")
        return
    addr = _addr(args[0])
    await cdp.evaluate(f"window.zxw.unwatch({addr})", return_by_value=False)
    print(f"Watchpoint removed at 0x{addr:04x}")


def cmd_help(_cdp, _args):
    print(__doc__)


COMMANDS = {
    "regs":      cmd_regs,
    "mem":       cmd_mem,
    "poke":      cmd_poke,
    "bp":        cmd_bp,
    "pause":     cmd_pause,
    "resume":    cmd_resume,
    "step":      cmd_step,
    "screen":    cmd_screen,
    "watch":     cmd_watch,
    "hits":      cmd_hits,
    "watch_add": cmd_watch_add,
    "watch_rm":  cmd_watch_rm,
    "help":      lambda c, a: print(__doc__),
}


# ── REPL ─────────────────────────────────────────────────────────────────────

async def repl(ws_url: str):
    print(f"Connecting to {ws_url}")
    async with websockets.connect(ws_url) as ws:
        cdp = CdpClient(ws)
        print("Connected. Type 'help' for commands, 'quit' to exit.")
        print()
        while True:
            try:
                line = input("zxw> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            parts = line.split()
            verb, args = parts[0].lower(), parts[1:]
            if verb in ("quit", "exit"):
                break
            handler = COMMANDS.get(verb)
            if handler is None:
                print(f"Unknown command '{verb}'. Type 'help' for a list.")
                continue
            try:
                result = handler(cdp, args)
                if asyncio.iscoroutine(result):
                    await result
            except RuntimeError as exc:
                print(f"JS error: {exc}")
            except Exception as exc:
                print(f"Error: {exc}")

    print("Disconnected.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9222
    ws_url = asyncio.run(_get_ws_url(port))
    asyncio.run(repl(ws_url))


if __name__ == "__main__":
    main()
