#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

MODE=all
SKIP_SYNC=0

usage() {
  cat <<'EOF'
Usage: ./scripts/check.sh [--skip-sync] [--quality|--runtime|--package]

With no mode, run the complete local package check. Focused modes are used by
CI after the job has prepared the locked environment.
EOF
}

while (($# > 0)); do
  case "$1" in
    --skip-sync)
      SKIP_SYNC=1
      ;;
    --quality|--runtime|--package)
      MODE="${1#--}"
      ;;
    --all)
      MODE=all
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

sync_environment() {
  if ((SKIP_SYNC)); then
    return
  fi

  # uv honours UV_PYTHON itself; otherwise an existing .venv keeps its
  # interpreter instead of being rebuilt around whatever `python` is on PATH.
  uv sync \
    --locked \
    --no-dev \
    --extra render-image \
    --extra adapters \
    --extra tui
}

ruff_command() {
  if command -v ruff >/dev/null 2>&1; then
    RUFF=(ruff)
  else
    RUFF=(uvx --from 'ruff>=0.11,<1' ruff)
  fi
}

pyright_command() {
  if command -v pyright >/dev/null 2>&1; then
    PYRIGHT=(pyright)
  else
    PYRIGHT=(uvx --from 'pyright>=1.1,<2' pyright)
  fi
}

python_command() {
  if [[ ! -x .venv/bin/python ]]; then
    echo ".venv/bin/python is missing; run without --skip-sync first" >&2
    exit 1
  fi
  PYTHON=(.venv/bin/python)
}

run_python() {
  "${PYTHON[@]}" "$@"
}

run_quality() {
  ruff_command
  "${RUFF[@]}" format --check src tests scripts
  "${RUFF[@]}" check src tests scripts

  pyright_command
  # Reads [tool.pyright] in pyproject.toml and resolves imports from the venv above.
  "${PYRIGHT[@]}"
}

run_runtime() {
  python_command
  # The editor tests skip themselves when textual is missing, so the gate
  # installs the extra and fails here instead of silently skipping them.
  run_python -c 'import textual'
  run_python -m unittest discover -s tests -v
  run_python -m compileall -q src tests scripts
}

run_package() {
  python_command
  rm -rf dist
  uv build
  uvx --from 'twine>=6,<7' twine check dist/*

  local wheel
  wheel=$(find dist -maxdepth 1 -type f -name '*.whl' -print -quit)
  if [[ -z "$wheel" ]]; then
    echo "wheel artifact not found" >&2
    exit 1
  fi
  run_python scripts/check_wheel.py "$wheel"

  local temporary
  temporary=$(mktemp -d)
  # Expand now: the local variable is gone when the trap fires at exit.
  trap "rm -rf '$temporary'" EXIT
  local python_executable
  python_executable=$(run_python -c 'import sys; print(sys.executable)')
  uv venv "$temporary/venv" --python "$python_executable"
  UV_LINK_MODE=copy uv pip install --python "$temporary/venv/bin/python" \
    "$wheel[adapters,render-image,tui]"

  # A fresh environment may resolve a different browser revision. Its renderer
  # must work too: a failed install is a failed package check, never a silent skip.
  "$temporary/venv/bin/playwright" install --with-deps chromium

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
}

export REQUIRE_RENDERER=1
sync_environment

case "$MODE" in
  quality)
    run_quality
    ;;
  runtime)
    run_runtime
    ;;
  package)
    run_package
    ;;
  all)
    run_quality
    run_runtime
    run_package
    ;;
  *)
    echo "unknown check mode: $MODE" >&2
    exit 2
    ;;
esac
