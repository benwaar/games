(module
  (memory (export "memory") 4)

  (func $pixel (param $x i32) (param $y i32)
    (i32.store
      (i32.add
        (i32.mul (local.get $y) (i32.const 1024))
        (i32.mul (local.get $x) (i32.const 4)))
      (i32.const 0xFFFFFFFF)))

  (func (export "render")
    (local $i i32)
    ;; fill black
    (local.set $i (i32.const 0))
    (block $done (loop $loop
      (br_if $done (i32.ge_u (local.get $i) (i32.const 262144)))
      (i32.store (local.get $i) (i32.const 0xFF000000))
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $loop)))
    ;; horizontal bar: y=128, x=108..148
    (local.set $i (i32.const 108))
    (block $h (loop $hl
      (br_if $h (i32.gt_u (local.get $i) (i32.const 148)))
      (call $pixel (local.get $i) (i32.const 128))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $hl)))
    ;; vertical bar: x=128, y=108..148
    (local.set $i (i32.const 108))
    (block $v (loop $vl
      (br_if $v (i32.gt_u (local.get $i) (i32.const 148)))
      (call $pixel (i32.const 128) (local.get $i))
      (local.set $i (i32.add (local.get $i) (i32.const 1)))
      (br $vl)))
  )

  (func (export "width") (result i32) (i32.const 256))
  (func (export "height") (result i32) (i32.const 256))
)
