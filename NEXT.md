# Next

Remaining work before starting Acoustic Odyssey or Artefact.

---

## 1. zxw — POK file support (Demo 2)

Study the real-hardware pattern for educational purposes: load a protected TAP, apply pokes
from a `.pok` file to bypass the startup check — the same technique players used with
Multiface hardware or POKE listings from magazines. The goal is understanding how binary
protection works, not breaking anything current.

Test subject: `j83.tap`. Three sysvar pokes bypass the startup protection check — without
them the game bails immediately. The `.pok` file is the bypass.

**M8** — Create `zxw/www/tap/j83.pok` with the three sysvar pokes  
Gate: file parses correctly and addresses match `j83-analysis.md`

**M9** — POK parser in Rust — parse `.pok` text format, return `Vec<(addr, value)>`  
Gate: unit tests pass for valid and malformed files

**M10** — Apply pokes on TAP load — after tape load, check for matching `.pok` by name,  
apply pokes before entry point runs  
Gate: `j83.tap` loads and reaches the loading screen with `j83.pok` applied

**M11** — CDP path — apply same pokes via the existing CDP debug API  
Gate: same result as M10, triggered via CDP

**M12** — Demo page — shows TAP name, POK file applied, addresses and values listed  
Gate: non-technical person can see what was poked and why
