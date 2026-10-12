PORT ?= 3500

.PHONY: build serve dev test clean assets deploy

build: assets
	wasm-pack build --target web --out-dir www/pkg

assets:
	@python3 -c "\
	import json, os; \
	m = {}; \
	[m.update({d: sorted(os.listdir('www/'+d))}) for d in ('rom','tap','scr') if os.path.isdir('www/'+d)]; \
	open('www/assets.json','w').write(json.dumps(m, indent=2))"
	@echo "assets.json updated"

test:
	cargo test

serve: build
	@echo "Serving at http://localhost:$(PORT)"
	cd www && python3 -m http.server $(PORT)

dev: build
	@lsof -ti :$(PORT) | xargs kill -9 2>/dev/null || true
	@echo "Serving at http://localhost:$(PORT)"
	cd www && python3 -m http.server $(PORT) &
	@sleep 1
	open http://localhost:$(PORT)

clean:
	cargo clean
	rm -rf www/pkg www/assets.json

deploy:
	@echo "Deploying incant output → zxw/www/tap/"
	@cp -v ../incant/output/*.tap www/tap/ 2>/dev/null || echo "  no TAPs in incant/output/"
	@$(MAKE) assets
