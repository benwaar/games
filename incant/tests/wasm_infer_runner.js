// WASM inference test runner — called by test_wasm_infer.py
// Usage: node wasm_infer_runner.js <wasm_path> <f32_hex>...
// Each <f32_hex> is 8 hex chars (4 bytes LE float32).
// Writes floats to WASM memory at offset 0, calls infer(0), prints i32 argmax.
const fs = require('fs');
const [,, wasmPath, ...hexFloats] = process.argv;
const wasmBytes = fs.readFileSync(wasmPath);
const floats = hexFloats.map(h => Buffer.from(h, 'hex').readFloatLE(0));

WebAssembly.instantiate(wasmBytes).then(({ instance }) => {
  const mem = new DataView(instance.exports.memory.buffer);
  floats.forEach((v, i) => mem.setFloat32(i * 4, v, true));
  process.stdout.write(String(instance.exports.infer(0)) + '\n');
}).catch(e => { process.stderr.write(e.message + '\n'); process.exit(1); });
