# Binary Protection — From Z80 Copy Protection to WASM Security

How software protection techniques evolved from 1983 Spectrum tape loaders to modern WebAssembly, and why the core ideas are the same.

## The core problem

You ship a binary. Someone else has the binary. You want to control what they can do with it — prevent copying, prevent reverse engineering, prevent tampering. The attacker has physical access to the bits. Every protection technique is a variation on making those bits harder to understand or modify.

## Z80 Spectrum techniques (1983–1990)

The Spectrum had no OS, no memory protection, no privilege levels. Everything ran in one flat 64K address space. Protection was pure ingenuity.

### System variable hijacking

The Spectrum ROM uses fixed memory addresses (0x5B00–0x5CB5) for housekeeping — error handlers, keyboard state, display parameters. Loader code writes directly into these addresses:

```z80
; Patch ERR_SP so BREAK jumps back into the game instead of BASIC
LD HL, game_restart
LD (5C78h), HL
```

A 1983 Spectrum game does this via `LOAD CODE` — it loads a 2-byte block directly to 0x5C78 (ERR_SP), redirecting the error handler. Press BREAK and you don't get BASIC — you get a crash or a restart.

**Why it works:** the ROM trusts the values in system variables unconditionally. There's no validation. If you put a new address in ERR_SP, that's where execution goes on error.

### Self-relocating code

Code is loaded at one address but designed to run at another. A small stub copies it into place at runtime:

```z80
LD HL, 6004h    ; source (loaded position + 4)
LD DE, 6000h    ; destination (intended position)
LD BC, 2000h    ; length
LDIR            ; copy byte-by-byte
JP 61E5h        ; jump to entry point
```

The stub itself is tiny and disposable. The real code only exists in its correct form for the instant between the copy completing and the first instruction executing.

**Why it works:** a static dump of the tape data doesn't give you the running program. You need to understand the relocation to reconstruct it.

### Multi-stage loading

Instead of one LOAD, the program loads in multiple stages. Each stage may decrypt or relocate the next. A typical 1983 game loads 5 blocks in a FOR loop — screen, code, stub, and two system variable patches.

**Why it works:** simple tape copiers that duplicate block-by-block may miss the ordering or the system variable writes. The whole sequence has to execute correctly.

### Checksum self-verification

The program computes a checksum of its own code at runtime and refuses to run if it's been modified:

```z80
LD HL, start_of_code
LD BC, length
XOR A
checksum_loop:
  XOR (HL)
  INC HL
  DEC BC
  LD A, B
  OR C
  JR NZ, checksum_loop
  CP expected_value
  JP NZ, crash
```

**Why it works:** any patch (to remove copy protection, for example) changes the checksum. The program detects tampering.

### Custom tape loaders

The Spectrum ROM has a standard tape loader. Games replaced it entirely — custom loaders with different pulse timings, different encoding schemes, different block structures. Standard tape copiers couldn't read them because they only understood the ROM's format.

**Why it works:** the protection is in the format, not the data. You need the custom loader to decode the tape.

## Worked example: protection-as-integrity-gate (1983 Spectrum game)

A 1983 Spectrum game uses all of the above techniques in a single loader. We rebuilt the TAP file, stripped the "copy protection" patches, and the game refused to start. Running the loader in a Z80 emulator (`tap_run_example.py`) confirmed why.

### What the loader does

Five blocks load in sequence:

1. **Screen** → 0x4000 (loading artwork)
2. **Game code** → 0x6000 (8192 bytes)
3. **Relocation stub** → 0x5B80 (15 bytes — LDIR + JP 61E5h)
4. **JP (HL) instruction** → 0x5CB0 (1 byte — traps BREAK)
5. **ERR_SP patch** → 0x5C78 (2 bytes — redirects error handler)

The stub relocates the game code (shifts 4 bytes left via overlapping LDIR), then jumps to 0x61E5.

### The integrity gate

The very first instructions at the entry point:

```z80
0x61E5: LD A, (5C79h)    ; read high byte of ERR_SP
0x61E8: CP 83h           ; expected value?
0x61EA: RET NZ           ; no → bail out to BASIC
```

Without the ERR_SP patch at 0x5C78, the comparison fails and `RET NZ` fires. The loading screen is already in screen memory, so you see the artwork — but the game never starts.

This is the key insight: **the protection IS the initialisation**. The developer didn't add a separate check that could be NOPed out. They wove it into the startup path. Stripping the "copy protection" doesn't unlock the game — it breaks it.

### Emulator confirmation

| Entry point | Sysvar patches | Result |
|-------------|---------------|--------|
| 0x61E5 | No | Bails at 0x61EA — RET NZ fires (ERR_SP check fails) |
| 0x61E5 | Yes | Gets past check, runs until it hits Spectrum ROM calls at 0x25D6 |
| 0x6000 | Either | Wrong entry point — skips init, hangs |

The fix: include the sysvar patches as POKE values in the BASIC loader. The rebuilt TAP (`example_fixed.tap`) does this. The game also requires Spectrum ROM routines (0x25D6 is in ROM space), so a full emulator (FUSE) is needed for complete execution.

### The modern parallel

This pattern appears in modern software licensing. A licence check that's tangled into the initialisation path — not a separate `if (licensed)` guard — is far harder to bypass. Remove it and the software crashes, not because of an intentional anti-tamper trap, but because the initialisation genuinely depends on the state the licence check establishes.

---

## WASM equivalents (2017–present)

WebAssembly has a real security model — sandboxed memory, validated bytecode, no direct code modification. But the same motivations exist: protect IP, prevent reverse engineering, enforce licensing.

### Function table corruption (↔ system variable hijacking)

WASM modules can have indirect call tables (`table` section). These are stored in linear memory or as runtime structures. An out-of-bounds write to linear memory can corrupt adjacent data that affects control flow:

```wat
;; Indirect call through table — if the table index is corrupted,
;; execution goes somewhere unexpected
(call_indirect (type $sig) (local.get $index))
```

**Research area:** in multi-module WASM applications, one module's memory corruption can affect another module's function tables if they share memory. This is the WASM equivalent of writing to system variables.

**Difference from Z80:** WASM validates table indices at runtime (trap on out-of-bounds). The Spectrum had no such check. But implementation bugs in WASM runtimes have been found that bypass this.

### Obfuscation (↔ self-relocating code)

WASM binary is already harder to read than JavaScript. On top of that, tools like `wasm-obfuscator` apply:

- **Control flow flattening** — replace structured if/else/loop with a state machine dispatch
- **Opaque predicates** — conditionals that always go one way but are hard to prove statically
- **Dead code injection** — add unreachable paths that confuse disassemblers
- **String encryption** — constants are decrypted at runtime

```wat
;; Flattened control flow — original logic is hidden in a state machine
(block $dispatch
  (block $state0
    (block $state1
      (block $state2
        (br_table $state0 $state1 $state2 $dispatch (local.get $state))
      ) ;; state2
      ;; ... actual logic scattered across states
    ) ;; state1
  ) ;; state0
)
```

**Why it works:** `wasm-decompile` and `wasm2wat` produce valid output, but the logic is incomprehensible. You see the instructions but not the intent — same as looking at a Z80 dump after relocation without understanding the stub.

### Dynamic module linking (↔ multi-stage loading)

WASM applications can load modules in stages:

```javascript
// Stage 1: load the shell
const shell = await WebAssembly.instantiateStreaming(fetch('shell.wasm'));

// Stage 2: shell decrypts and loads the real module
const decrypted = shell.instance.exports.decrypt(encrypted_bytes);
const real = await WebAssembly.instantiate(decrypted, imports);
```

The shipped `.wasm` isn't the running code. It's a loader that produces the running code at runtime. Identical concept to the Spectrum's multi-stage loaders.

**Why it works:** static analysis of the shipped binary doesn't reveal the real logic. You need to run the loader.

### Subresource integrity (↔ checksum self-verification)

Browsers can verify WASM modules haven't been tampered with:

```html
<script src="app.js"
  integrity="sha384-oqVuAfXRKap7fdgcCY5uykM6+R9GqQ8K/uxy9rx7HNQlGYl1kPzQho1wx4JwY8w"
  crossorigin="anonymous">
</script>
```

And at the application level, a WASM module can hash its own memory or imported modules:

```wat
;; Compute hash of linear memory region
(func $verify_integrity (param $start i32) (param $len i32) (result i32)
  ;; XOR-based integrity check (same concept as the Z80 version)
  ...
)
```

**Difference from Z80:** SRI is enforced by the browser, not the code itself. The Z80 had to self-check because there was no trusted third party.

### Anti-debugging (↔ anti-BREAK)

WASM modules can detect developer tools:

- **Timing checks** — measure execution time of a known-cost operation. If it takes too long, a debugger is stepping through
- **DevTools detection** — check for `window.devtoolsOpen` or similar signals from the JS bridge
- **Control flow traps** — set breakpoint-sensitive patterns that behave differently under debugging

```javascript
// JS bridge: detect if DevTools are open
const start = performance.now();
debugger; // This pauses only if DevTools are open
const elapsed = performance.now() - start;
if (elapsed > 100) {
  // Debugger detected — refuse to run, corrupt state, or phone home
}
```

**Same arms race as the Spectrum.** The Spectrum community had BREAK-key disablers; crackers had POKEs to re-enable them. WASM has anti-debug; reverse engineers have instrumented runtimes that lie about timing.

### JS bridge attacks (no Z80 equivalent)

This is new to WASM. The sandbox is strong, but the glue code between WASM and JavaScript is often the weak point:

```javascript
// WASM exports a function pointer. JS trusts it.
const result = wasmInstance.exports.process(userInput);
// If 'process' returns a pointer into linear memory, and JS reads
// it without bounds checking, the WASM module controls what JS sees.
```

The Z80 had no equivalent because there was no sandbox to bridge. Everything was flat memory. The WASM innovation is the sandbox — and the attack surface is where the sandbox meets the unsandboxed world.

### Supply chain opacity (↔ custom tape loaders)

A `.wasm` binary is opaque. Unlike JavaScript (which is source text), WASM ships as bytecode. You can decompile it, but:

- Variable names are stripped
- Structure is lost
- Optimisation has transformed the logic
- Obfuscation makes it worse

This means a dependency you `npm install` could include a `.wasm` blob that does anything. The equivalent of a Spectrum game with a custom loader — you can't tell what it does without reverse engineering it.

**Active concern:** WASM is increasingly used in npm packages for performance-critical code. A supply chain attack via `.wasm` is harder to detect than one via `.js`.

## The arms race pattern

Every generation follows the same cycle:

```
1. Platform ships with no protection     → Spectrum ROM, early WASM
2. Creators add protection               → Tape loaders, obfuscation
3. Attackers develop analysis tools       → Tape copiers, wasm-decompile
4. Creators add anti-analysis             → Self-checks, anti-debug
5. Attackers develop better tools         → Emulator-based crackers, instrumented runtimes
6. Platform adds native security          → (Spectrum never got here), WASM SRI + sandbox
7. Attackers find gaps in the platform    → N/A, JS bridge attacks + side channels
```

The Z80 Spectrum got to step 5. WASM is at step 7. The concepts are the same — the execution environment is more sophisticated.

## CDP as a Spectrum debugger — live WASM memory patching

Chrome DevTools Protocol gives an attacker (or a security tester) the same capabilities as a Spectrum emulator's debugger — pause execution, inspect memory, patch values, resume. The difference is this works on production web apps.

### The POK file parallel

On the Spectrum, POK files patch memory addresses after a game loads:

```
; 1983 Spectrum game — infinite lives
POKE 24698, 0
```

Via CDP, the same thing works on any WASM module running in a browser:

```javascript
// CDP Runtime.evaluate — reach into WASM linear memory
const mem = new Uint8Array(wasmInstance.exports.memory.buffer);

// Read a value (inspect)
console.log(mem[0x1A3F]);

// Write a value (patch)
mem[0x1A3F] = 0xFF;  // "infinite lives" — or bypass a licence check
```

### What CDP exposes

| CDP capability | Spectrum equivalent | What it means for WASM security |
|---------------|--------------------|---------------------------------|
| `Runtime.evaluate` | POKE command | Write any value to any linear memory address |
| `Debugger.pause` + inspect | BREAK + register dump | Pause WASM execution, inspect locals, value stack, memory |
| `Debugger.setBreakpoint` | Breakpoint in STS debugger | Stop at a specific WASM function, inspect state before/after |
| `Network.setRequestInterception` | Swapping tapes | Replace the `.wasm` binary before it loads — patch the module, not just memory |
| `Runtime.evaluate` (call exports) | `USR` / `RANDOMIZE USR` | Call any exported WASM function directly, bypass the UI |
| Full memory dump | Spectrum memory viewer | Read the entire linear memory — scan for patterns, map data structures |

### Worked example: bypassing a WASM licence check

A WASM module stores a licence flag in linear memory:

```wat
;; Licence check — reads a byte from memory offset 0x0100
(func $check_licence (result i32)
  (i32.load8_u (i32.const 256))  ;; 0 = unlicensed, 1 = valid
)

(func $start_app
  (if (i32.eqz (call $check_licence))
    (then (call $show_paywall))
    (else (call $show_dashboard))
  )
)
```

Attack via CDP:

```javascript
// 1. Wait for module to instantiate
// 2. Patch the licence byte
const mem = new Uint8Array(wasmInstance.exports.memory.buffer);
mem[256] = 1;  // licensed = true

// 3. Or call the function directly, bypassing the check entirely
wasmInstance.exports.show_dashboard();
```

This is the 1983 ERR_SP patch in modern form. The game checks a memory address; the attacker writes the expected value. Forty years apart, same vulnerability.

### Defence implications

The lesson is not "don't use WASM" — it's **never trust linear memory for security decisions**:

| Vulnerable pattern | Why it fails | What to do instead |
|-------------------|-------------|-------------------|
| Licence/auth flag in linear memory | Pokeable via CDP or any JS with access to the module | Verify server-side on every protected action |
| Feature gates stored as memory bytes | Scannable and patchable | Server-side feature flags; WASM module only receives what it's allowed to use |
| Crypto keys in linear memory | Dumpable via `memory.buffer` | Use Web Crypto API (keys never enter JS/WASM address space) |
| Integrity self-check in WASM | Attacker patches the check too, or intercepts the module before load | SRI hash on the `<script>` tag + server-side validation of results |
| Anti-debug timing checks | CDP can override `performance.now()` | Assume debugging is always possible; don't rely on client-side detection |

**The rule:** anything in the browser is the user's machine. WASM's sandbox protects the *host* from the *module*, not the module from the user. The Spectrum taught this in 1983 — every copy protection scheme was broken because the user had the hardware. CDP is the modern equivalent of having the hardware.

## WASM memory proposals — the old attacks come back

The Spectrum had one thread, one flat address space, no protection. We spent 40 years adding protection layers. Recent WASM proposals are selectively removing them again — for performance. The trade-off reopens attack surfaces that C/C++ security researchers spent two decades learning to defend against.

### What's shipping now

| Feature | Browser support | What it changes | Security surface |
|---------|----------------|-----------------|-----------------|
| **Threads + SharedArrayBuffer** | Chrome, Firefox, Safari | Shared linear memory between workers. Multiple execution contexts reading and writing the same bytes concurrently | Race conditions, TOCTOU, double-free patterns, Spectre timing channels. SharedArrayBuffer was disabled in all browsers for 2+ years after Spectre (2018–2020) because shared memory gave attackers a high-resolution timer. Re-enabled with site isolation mitigations, but the underlying problem — shared memory = timing channel — hasn't gone away |
| **Memory64** | Chrome, Firefox | 64-bit address space for linear memory. 4GB limit removed | Larger heap = more room for heap spraying. Bounds checking on 64-bit addresses is more complex. Address space layout is harder to predict — ASLR-style defences become relevant |
| **GC (garbage collection)** | Chrome, Firefox | Managed reference types that cross the JS/WASM boundary. WASM can create and hold GC-managed objects | New object types at the JS↔WASM bridge. Type confusion between managed and unmanaged references. GC pressure as a side channel |
| **Bulk memory operations** | All browsers | `memory.copy`, `memory.fill` — fast bulk writes to linear memory | Makes memory corruption faster. A single `memory.fill` can zero out or overwrite large regions. Heap spray at native speed |

### What's coming

| Feature | Phase | What it opens up |
|---------|-------|-----------------|
| **Shared-Everything Threads** | Phase 1 | Not just shared memory — shared *tables, globals, functions* across threads. Every cross-thread attack surface multiplied. One thread can modify another thread's function table entries mid-execution |
| **Memory Control** | Phase 1 | Fine-grained memory management — `mprotect`-style page permissions. Could be defensive (mark pages read-only) or offensive (change permissions at runtime to bypass W^X) |
| **Custom Page Sizes** | Phase 3 | Non-default memory page sizes. Allocator assumptions about page alignment break. Off-by-one in page boundary calculations = potential sandbox escape |
| **Acquire-Release Atomics** | Phase 2 | Weaker-than-sequential-consistency memory ordering. Correct use prevents races; incorrect use creates subtle concurrency bugs that are exploitable and nearly impossible to reproduce in testing |

### The classic attacks that apply again

Threads + SharedArrayBuffer is the proposal that reopens the old playbook. Shared mutable memory between execution contexts is exactly what made C/C++ exploits possible:

**Race conditions (TOCTOU)**

The Spectrum had no concurrency — one CPU, one thread. WASM with shared memory has the classic time-of-check-to-time-of-use problem:

```javascript
// Thread A: security check
if (sharedMem[AUTH_OFFSET] === VALID_TOKEN) {
  // Thread B writes 0 to AUTH_OFFSET right here
  performSensitiveAction();  // auth check passed but token is now invalid
}
```

This is the multi-threaded version of the 1983 ERR_SP patch. Instead of poking a value after load, you poke it between check and use.

**Heap spray**

Memory64 gives a 64-bit address space. Bulk memory ops give fast fills. Together:

```wat
;; Fill 1GB of linear memory with a NOP sled + payload address
;; Equivalent of classic heap spray, but at WASM speed
(memory.fill
  (i32.const 0)           ;; destination
  (i32.const 0x90909090)  ;; NOP sled pattern
  (i32.const 1073741824)  ;; 1GB
)
```

On the Spectrum, filling 48K of RAM took noticeable time. In WASM, filling gigabytes is a single instruction.

**Spectre timing**

SharedArrayBuffer gives a high-resolution timer by incrementing a shared counter in a worker loop:

```javascript
// Worker: increment shared counter as fast as possible
const counter = new Int32Array(sharedBuffer);
while (true) Atomics.add(counter, 0, 1);

// Main thread: use counter as a high-res timer for cache timing attacks
const start = Atomics.load(counter, 0);
// ... speculative access ...
const elapsed = Atomics.load(counter, 0) - start;
```

This was the original reason browsers disabled SharedArrayBuffer post-Spectre. Site isolation mitigates the cross-origin case, but same-origin timing attacks remain possible.

**Function table corruption (Shared-Everything Threads)**

When tables are shared across threads, one thread can modify another's indirect call targets:

```wat
;; Thread A expects table[0] = $safe_function
(call_indirect (type $sig) (i32.const 0))

;; Thread B overwrites table[0] with $malicious_function
;; If the timing is right, Thread A calls the wrong function
(table.set (i32.const 0) (ref.func $malicious_function))
```

This is the WASM version of patching Spectrum system variables — but concurrent, making it non-deterministic and much harder to detect.

### Defence posture

The defensive takeaway isn't "avoid these features" — it's "design systems that assume shared memory is hostile":

| Old C/C++ defence | WASM equivalent | Status |
|-------------------|-----------------|--------|
| ASLR (randomise addresses) | WASM linear memory starts at a fixed offset (0) — no ASLR | Not available. Memory layout is predictable by design |
| Stack canaries | No equivalent in WASM — stack is managed by the engine, not in linear memory | Engine-level protection only. User-space canaries can be bypassed |
| W^X (write XOR execute) | WASM bytecode is immutable after compilation. But function tables and linear memory are mutable | Partial. Code is safe; data used for control flow (tables, indirect calls) is not |
| Thread sanitiser (TSan) | No WASM equivalent in browsers | Gap. Race conditions in WASM shared memory are largely undetectable by browser tooling |
| Sandboxing (seccomp, pledge) | WASM sandbox is strong for host protection. No equivalent for module-to-module isolation within shared memory | The Component Model (Phase 1) aims to fix this with capability-based isolation |

**The fundamental problem:** WASM's security model was designed for single-threaded, isolated linear memory. Every shared-memory proposal weakens that original isolation guarantee. The mitigations (site isolation, atomic operations, structured concurrency) are bolted on — not inherent to the model. This is the same arc as C → C with threads → C with threads and shared libraries → decades of CVEs.

## Architecture decision: how much goes in WASM?

The proposals above don't affect all WASM usage equally. The risk depends on *what you put in the module*.

### Data viz component vs full application

| | Data viz component in WASM | Full UI in WASM (e.g. Flutter web) |
|--|---|---|
| **What's in linear memory** | Chart data, rendering state, layout calculations | Auth state, session tokens, route guards, business logic, form validation, feature flags, API credentials — the entire application |
| **Attack surface** | Poke the chart data. Worst case: wrong numbers on screen | Poke auth state → bypass login. Disable route guards → access admin pages. Extract tokens → impersonate user |
| **JS bridge width** | Narrow: host passes data in, gets pixels out | Wide: WASM module talks to every browser API through the bridge. Every bridge call is an interception point |
| **If memory is dumped** | Chart coordinates, colour values | The application's complete runtime state — everything a server trusted the client with |
| **Shared memory risk** | Low — rendering threads share pixel buffers | High — auth checks, feature flags, and business rules share memory with UI threads |

### The Spectrum parallel

A 1983 Spectrum game put its copy protection (ERR_SP patch, system variable writes) in the same flat 64K as the game itself. Crackers didn't need to break the protection scheme — they just needed to find the address and write a byte. The protection lived in the same address space as the thing it was protecting.

Flutter web (and any "compile the whole app to WASM" framework) does the same thing. Auth logic, route guards, feature gates, and business rules all compile into the same linear memory as the UI rendering. An attacker with CDP doesn't need to break the auth scheme — they just need to find the right offset and write a byte. Or call an exported function. Or intercept the module at load time.

The traditional web model avoids this by keeping sensitive decisions out of the client entirely:

```
Traditional web:
  Server: auth, routing, business logic, data access
  Client: rendering, interaction, display
  → sensitive state never enters the client's address space

Flutter web / full-WASM app:
  Server: data access (maybe)
  Client: auth, routing, business logic, rendering — all in one linear memory
  → sensitive state is in the same flat address space as the UI
  → every byte is readable and writable via CDP
```

This doesn't mean WASM is wrong for UI — it means the architecture decision matters. A contained component (data viz, media processing, computation) has a narrow attack surface. A full application framework that compiles everything into one WASM module has the widest possible surface — and with shared memory proposals, it gets wider.

### Practical guidance

| Use case | WASM risk | Recommendation |
|----------|-----------|----------------|
| **Data viz / charting** | Low — worst case is cosmetic | Good fit. Fast rendering, contained scope, narrow bridge |
| **Media processing** | Low — audio/video pipelines, no auth state | Good fit. Performance-critical, isolated data |
| **Computation** (ML inference, crypto, compression) | Low — pure functions, no application state | Good fit. This is what WASM was designed for |
| **Full application UI** (Flutter web, Yew, Leptos) | High — entire app state in linear memory | Caution. Auth, routing, and business logic should stay server-side. Use WASM for rendering only, not decisions |
| **Full app + shared memory / threads** | Very high — concurrent access to application state | Strong caution. Classic C/C++ concurrency attack surface reintroduced at the application level |

The question isn't "is WASM secure?" — it's "how much of your application's security-sensitive logic lives in a flat, inspectable, patchable address space?" The less, the better. The Spectrum taught this in 1983.

## Building it: Z80 self-verifying binary (M9 PoC)

The theory above describes checksum self-verification. Here's the working implementation — a 26-byte prefix that protects any Z80 binary.

### The verifier prefix

```z80
; 26 bytes — prepended to any payload binary
  LD HL, payload_start   ; 21 lo hi   — start of protected region
  LD BC, payload_len     ; 01 lo hi   — bytes to check
  LD E, 0                ; 1E 00      — XOR accumulator
loop:
  LD A, (HL)             ; 7E         — load next byte
  XOR E                  ; AB         — accumulate
  LD E, A                ; 5F         — store back
  INC HL                 ; 23
  DEC BC                 ; 0B
  LD A, B                ; 78         — check BC == 0
  OR C                   ; B1
  JR NZ, loop            ; 20 F7      — keep going
  LD A, E                ; 7B         — get final checksum
  CP expected            ; FE xx      — compare to patched value
  JP NZ, crash           ; C2 lo hi   — tampered → halt
  JP payload_start       ; C3 lo hi   — intact → run
```

The trick: the `expected` byte (offset 19) is patched **after** the payload is assembled. The gate computes `XOR` of all payload bytes, writes that value into the prefix, and the binary is ready.

### Memory layout

```
load_addr:
  +0   .. +25     verifier prefix (26 bytes)
  +26  .. +26+N-1 payload (N bytes — the real program)
  +26+N           HALT (crash target — verifier jumps here on tamper)
```

### What the gate does

```python
from incant.protect import wrap_z80_checksum

payload = bytes([0x3E, 0x42, 0x76])  # LD A, 0x42; HALT
protected = wrap_z80_checksum(payload, load_addr=0x8000)
# protected is 30 bytes: 26-byte prefix + 3-byte payload + 1 crash HALT
# The checksum byte at offset 19 is automatically correct
```

### Test results

| Scenario | Result |
|----------|--------|
| Unmodified binary | Verifier passes → payload runs → A = 0x42 |
| Flip any payload byte | Checksum mismatch → JP to crash HALT |
| Flip every byte individually | All detected — zero false negatives |

### Attacker bypasses (Z80)

**Bypass 1 — NOP the branch:** replace `JP NZ, crash` (3 bytes at offset 20) with three NOPs. The verifier still runs but the branch never fires. Cost: find the JP NZ opcode (0xC2) in the prefix, write 0x00 0x00 0x00.

**Bypass 2 — Fix the checksum:** after patching the payload, recompute the XOR and write it to offset 19. The verifier passes because the checksum matches the tampered code. Cost: know the checksum algorithm (XOR is trivial to reverse).

**Why XOR is weak:** XOR checksum is linear — `XOR(a) ⊕ XOR(b) = XOR(a ⊕ b)`. An attacker who flips byte X can flip another byte to compensate, keeping the same checksum. A cryptographic hash (SHA, CRC-32) would be stronger but costs more Z80 cycles.

---

## Building it: WASM self-verifying module (M9 PoC)

WASM can't read its own code section — the bytecode is separate from linear memory (Harvard architecture). So the protection technique is different: store a known-good copy of critical data in a `(data ...)` segment, then hash that region at startup.

### The WAT module

```wat
(module
  (memory (export "memory") 1)
  (data (i32.const 1024) "\de\ad\be\ef")  ;; known-good payload

  (func $verify (result i32)
    (local $i i32) (local $end i32) (local $acc i32)
    (local.set $i (i32.const 1024))
    (local.set $end (i32.const 1028))
    (local.set $acc (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_u (local.get $i) (local.get $end)))
        (local.set $acc
          (i32.xor (local.get $acc) (i32.load8_u (local.get $i))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))
    (local.get $acc))

  (func (export "run") (result i32)
    (if (result i32) (i32.ne (call $verify) (i32.const 97))
      (then (unreachable))          ;; tampered → trap
      (else (i32.const 42)))))      ;; intact → return 42
```

### The Harvard constraint

On Z80, code and data share the same 64K — the verifier reads its own bytes with `LD A, (HL)`. In WASM, the code section is immutable and inaccessible from linear memory. You can only verify data segments, not the compiled functions themselves.

This means WASM self-verification protects **data integrity** (weights, keys, configuration), not **code integrity**. An attacker who modifies the WASM bytecode (function bodies, control flow) isn't caught by a data-segment hash.

### Attacker bypasses (WASM)

**Bypass 1 — Patch the comparison constant:** find the `i32.const 97` (the expected checksum) in the compiled WASM binary and change it to match the tampered data. In the binary, `i32.const` is opcode `0x41` followed by an LEB128-encoded value.

**Bypass 2 — Intercept at load time:** replace the `.wasm` binary before the browser loads it (CDP `Network.setRequestInterception`). The module never sees its original checksums.

**Bypass 3 — Patch linear memory via CDP:** after the module loads, write directly to the memory region the verifier checked. The check already passed; the data changes afterward.

### The real lesson

Data-segment verification catches accidental corruption and naive patching. It does NOT stop a motivated attacker who controls the runtime. The same conclusion as the Z80 case — but WASM's Harvard split makes it even weaker, because you can't verify the code at all.

## Key takeaway

The Spectrum is a perfect lab for this because the entire system fits in your head. 64K of memory, a few hundred opcodes, no OS. You can understand every byte. The protection techniques are the same as modern ones — just without the complexity of a browser runtime, an OS, and a network stack on top.

The M9 PoC (`incant/protect.py`) makes this concrete: the same XOR checksum technique, implemented on both Z80 and WASM, with working bypass code for each. The Z80 version is stronger (verifies actual code bytes); the WASM version is weaker (can only verify data segments). Both are trivially broken by someone who understands the algorithm.

The defensive insight: understanding the attack surface is the only way to protect against it. Every technique in this document — from the 1983 ERR_SP hijack to CDP memory patching — exists because the defender didn't account for what the attacker could reach. The Spectrum teaches this in miniature. WASM and CDP are the same lesson at production scale.

---

## The WASM security model points the wrong way

This is the uncomfortable finding from M9. It's not that WASM is broken — it's that WASM's security was designed for a threat model that doesn't match what most developers assume.

### What WASM actually protects

WASM's sandbox protects the **host** from the **module**:

- A malicious `.wasm` can't read your filesystem
- It can't make system calls the host didn't import
- It can't escape linear memory into the browser's address space
- It can't corrupt another tab's memory

This is real, valuable security. It's the reason you can run untrusted WASM in a browser without worrying about it installing malware. The sandbox is strong and well-tested.

### What WASM doesn't protect

WASM does **not** protect the **module** from the **host**:

- The user (or anyone with CDP access) can read the entire linear memory
- They can write to any address — auth flags, licence checks, feature gates
- They can call any exported function directly, bypassing the intended UI flow
- They can intercept and replace the `.wasm` binary before it loads
- They can modify the compiled bytecode in transit or on disk

And critically, from the M9 PoC: **a WASM module cannot verify its own code**. The Harvard architecture split means compiled functions are opaque to the module itself. There is no `LD A, (HL)` equivalent — no way for a function to read its own bytecode from linear memory.

### Why this matters more than it sounds

On the Z80, the same limitation doesn't exist. The Spectrum's flat memory model means code can read and checksum itself. The 1983 game's ERR_SP check works because the entry point can inspect system variables that the loader wrote. The entire address space is one flat surface — code, data, system state, all readable.

WASM deliberately broke that. Code and data are separated. The module can hash its data segments, but an attacker who patches the *verification function itself* (changing `i32.ne` to `i32.eq`, or `unreachable` to `nop`) bypasses the check entirely — and the module has no way to detect it.

This means the protection ceiling on WASM is **lower** than on a 1983 8-bit computer:

| Capability | Z80 Spectrum | WASM |
|-----------|-------------|------|
| Code can read own bytes | Yes — `LD A, (HL)` anywhere in 64K | No — code section is opaque |
| Code can checksum itself | Yes — XOR/CRC over own address range | No — can only checksum data segments |
| Attacker can patch code | Yes — same flat memory | Yes — modify `.wasm` binary or intercept at load |
| Module detects code patch | Possible — self-checksum catches it | Impossible — no access to own bytecode |
| Attacker can patch data | Yes — POKE | Yes — CDP / `memory.buffer` |
| Module detects data patch | Yes — checksum | Yes — but attacker can patch the check too |

The Z80 defender has strictly more tools than the WASM defender. The only advantage WASM has is SRI (Subresource Integrity) — a browser-enforced hash of the module at load time. But SRI verifies the file on the wire, not the module in memory. Post-load, the module is on its own, and it can't even see its own code.

### The real-world consequence

This affects every architecture that puts security-sensitive logic in WASM:

- **Licence checks in WASM** — unenforceable client-side. The check function itself can be patched. Server-side verification is the only option that works.
- **Feature gates** — if the decision is "does the user have access to feature X?" and the answer lives in linear memory or a WASM function, it's a single byte patch away from "yes".
- **Anti-tamper** — the strongest technique available (data-segment checksums) is a speed bump. The attacker patches the data *and* the checksum constant *and* the branch, in three edits.
- **DRM / content protection** — WASM cannot protect content from the user who is running it. The decryption key must enter linear memory at some point, and at that point it's readable.

### What WASM security is actually good for

The inversion: WASM excels at protecting the **server** from the **client**, not the other way around.

- Run untrusted user-submitted code in a WASM sandbox (serverless functions, plugin systems) — strong isolation
- Process sensitive data in a WASM module that can't exfiltrate it (no network, no filesystem) — capability-based security
- Run ML inference on-device without exposing model architecture to a malicious plugin — the sandbox keeps the plugin out

These are all cases where the host doesn't trust the module. That's what WASM was built for. The problems start when developers flip the trust relationship and try to use WASM to protect the module from the host — that's the Spectrum copy-protection problem all over again, with fewer tools.

### The Spectrum lesson

In 1983, every Spectrum game's copy protection was broken. Not because the protections were bad — some were genuinely clever (integrity gates, Speedlock's custom loaders). They were broken because the attacker had the hardware. You can't protect software from someone who controls the machine it runs on.

WASM is the same machine. The browser is the user's hardware. CDP is the modern POKE. The sandbox protects the host — and only the host. Forty years later, the fundamental constraint hasn't changed: **if the user has the binary, they have the binary**.
