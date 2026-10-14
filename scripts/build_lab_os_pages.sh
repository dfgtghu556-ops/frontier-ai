#!/usr/bin/env bash
# Rebuild the public copy of the Lab OS dashboard (GitHub Pages serves docs/ of the project branch):
#   https://dfgtghu556-ops.github.io/frontier-ai/dashboard/
# Run after scripts/export_lab_state.py whenever the records change, then commit docs/dashboard/.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/apps/lab-os"
[ -d node_modules ] || npm ci --no-audit --no-fund
rm -rf dist
LAB_OS_BASE=/frontier-ai/dashboard/ npm run build >/dev/null
rm -rf "$root/docs/dashboard"
cp -r dist/client "$root/docs/dashboard"
rm -rf dist
echo "[lab-os-pages] docs/dashboard rebuilt from $(git -C "$root" log -1 --format=%h) + working tree"
