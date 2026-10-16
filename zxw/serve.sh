#!/bin/bash
# Serve zxw with Cross-Origin-Isolation headers (required for SharedArrayBuffer).
cd "$(dirname "$0")/www" && python3 -c "
import http.server, functools

class COIHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

http.server.HTTPServer(('', 8998), COIHandler).serve_forever()
"
