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

Jetpac does this via `LOAD CODE` — it loads a 2-byte block directly to 0x5C78 (ERR_SP), redirecting the error handler. Press BREAK and you don't get BASIC — you get a crash or a restart.

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

Instead of one LOAD, the program loads in multiple stages. Each stage may decrypt or relocate the next. Jetpac loads 5 blocks in a FOR loop — screen, code, stub, and two system variable patches.

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

## Key takeaway

The Spectrum is a perfect lab for this because the entire system fits in your head. 64K of memory, a few hundred opcodes, no OS. You can understand every byte. The protection techniques are the same as modern ones — just without the complexity of a browser runtime, an OS, and a network stack on top. Learn them here, apply them to WASM.
