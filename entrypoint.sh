#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cleanup: kill all child and known grand-child processes on any exit
# -----------------------------------------------------------------------------
cleanup() {
  echo "→ Cleaning up all child processes…"
  # kill direct children
  pkill -P $$ || true
  # also kill any leftover vLLM servers
  pkill -f 'vllm\.entrypoints\.openai\.api_server' || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# 1) Ensure no stale API, sim or vLLM processes are running
# -----------------------------------------------------------------------------
echo "→ Killing any old api.py, simulation_env.py or vLLM processes…"
pkill -f 'python .*/api_server/api.py'            || true
pkill -f 'python .*/env_setup/simulation_env.py'  || true
pkill -f 'vllm\.entrypoints\.openai\.api_server'  || true

# wait up to 5s for port 8001 to free
for i in {1..10}; do
  if ! ss -ltnp 2>/dev/null | grep -q ':8001'; then
    break
  fi
  sleep 0.5
done

# -----------------------------------------------------------------------------
# 2) Start the Flask API (api_server/api.py)
# -----------------------------------------------------------------------------
echo "→ Starting api.py…"
pushd api_server >/dev/null
python api.py &
API_PID=$!
popd >/dev/null

# wait for port 8000 to be listening
for i in {1..10}; do
  if ss -ltnp 2>/dev/null | grep -q ':8000'; then
    echo "→ api.py is up"
    break
  fi
  sleep 0.5
done

# -----------------------------------------------------------------------------
# 3) Start the simulation environment (env_setup/simulation_env.py)
# -----------------------------------------------------------------------------
echo "→ Starting simulation_env.py…"
pushd env_setup >/dev/null
python simulation_env.py &
SIM_PID=$!
popd >/dev/null

sleep 1

# -----------------------------------------------------------------------------
# 4) Run the GRPO trainer (trainer/grpo.py) in foreground
# -----------------------------------------------------------------------------
echo "→ Starting grpo.py…"
pushd trainer >/dev/null
python grpo.py
GRPO_EXIT=$?
popd >/dev/null

# -----------------------------------------------------------------------------
# 5) On exit, kill API and sim (cleanup trap will also kill vLLM)
# -----------------------------------------------------------------------------
echo "→ grpo.py exited with code $GRPO_EXIT; shutting down background processes…"
kill $API_PID $SIM_PID || true

exit $GRPO_EXIT
