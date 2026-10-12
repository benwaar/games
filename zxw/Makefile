.PHONY: build serve

build:
	wasm-pack build --target web --out-dir www/pkg

serve:
	cd www && python3 -m http.server 8080
