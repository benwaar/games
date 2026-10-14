#!/bin/bash
cd "$(dirname "$0")/www" && python3 -m http.server 8998
