#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

uv sync --locked --no-dev --extra render-image --extra adapters --extra tui
export REQUIRE_RENDERER=1
if command -v ruff >/dev/null 2>&1; then
  RUFF=(ruff)
else
  RUFF=(uvx --from 'ruff>=0.11,<1' ruff)
fi
"${RUFF[@]}" format --check src tests scripts
"${RUFF[@]}" check src tests scripts
if command -v pyright >/dev/null 2>&1; then
  PYRIGHT=(pyright)
else
  PYRIGHT=(uvx --from 'pyright>=1.1,<2' pyright)
fi
# Reads [tool.pyright] in pyproject.toml and resolves imports from the venv above.
"${PYRIGHT[@]}"
UV_RUN=(uv run --no-dev --locked --extra render-image --extra adapters --extra tui)
# The editor tests skip themselves when textual is missing, so the gate installs
# the extra and fails here instead of silently skipping them.
"${UV_RUN[@]}" python -c 'import textual'
"${UV_RUN[@]}" python -m unittest discover -s tests -v
"${UV_RUN[@]}" python -m compileall -q src tests scripts

rm -rf dist
uv build
uvx --from 'twine>=6,<7' twine check dist/*

wheel=$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)
if [[ -z "$wheel" ]]; then
  echo "wheel artifact not found" >&2
  exit 1
fi
"${UV_RUN[@]}" python scripts/check_wheel.py "$wheel"

temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
python_executable=$("${UV_RUN[@]}" python -c 'import sys; print(sys.executable)')
uv venv "$temporary/venv" --python "$python_executable"
UV_LINK_MODE=copy uv pip install --python "$temporary/venv/bin/python" \
  "$wheel[adapters,render-image,tui]"

# A fresh environment may resolve a different browser revision. Its renderer
# must work too: a failed install is a failed package check, never a silent skip.
"$temporary/venv/bin/playwright" install chromium

mkdir -p "$temporary/profiles"
cat > "$temporary/profiles/smoke.json" <<'JSON'
{
  "protocol": "openai-responses",
  "model": "smoke-model"
}
JSON
cat > "$temporary/plugin.env" <<EOF
AGENT_CHAT_PROFILE_DIR=$temporary/profiles
AGENT_CHAT_DATA_DIR=$temporary/data
AGENT_CHAT_DEFAULT_PROFILE=smoke
AGENT_CHAT_CLEANUP_INTERVAL_SECONDS=0
AGENT_CHAT_MESSAGE_SEND_DELAY_SECONDS=0
EOF

(
  cd "$temporary"
  AGENT_CHAT_ENV_FILE="$temporary/plugin.env" \
    "$temporary/venv/bin/python" "$ROOT/scripts/smoke_plugin.py"
  "$temporary/venv/bin/nonebot-agent-chat" \
    --no-env --profiles-dir "$temporary/profiles" --check
  "$temporary/venv/bin/python" -m nonebot_plugin_agent_chat --help >/dev/null
)

(
  cd "$ROOT"
  "$temporary/venv/bin/python" -m unittest discover -s tests -q
)

echo "package checks: ok"
