# #!/usr/bin/env bash
# set -euo pipefail

# echo "→ Starting api.py…"
# cd api_server
# python api.py &
# API_PID=$!

# # (Optional) wait for the API to finish booting before kicking off the next script
# sleep 3
# cd ..

# echo "→ Starting simulation_env.py…"
# cd env_setup
# python simulation_env.py &
# SIM_PID=$!


# # (Optional) wait for your simulator to initialize
# sleep 3
# cd ..

# echo "→ Starting grpo.py…"
# cd trainer
# python grpo.py
# GRPO_EXIT=$?

# echo "→ grpo.py exited with code $GRPO_EXIT; shutting down background processes…"
# kill $API_PID $SIM_PID || true

# exit $GRPO_EXIT

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Cleanup: kill all child processes on any exit (error, Ctrl-C, or normal)
# -----------------------------------------------------------------------------
cleanup() {
  echo "→ Cleaning up child processes…"
  pkill -P $$ || true
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# 1) Ensure no stale API or sim processes are running
# -----------------------------------------------------------------------------
echo "→ Killing any old api.py or simulation_env.py processes…"
pkill -f 'python .*/api_server/api.py'   || true
pkill -f 'python .*/env_setup/simulation_env.py' || true

# -----------------------------------------------------------------------------
# 2) Start the Flask API (api_server/api.py)
# -----------------------------------------------------------------------------
echo "→ Starting api.py…"
pushd api_server >/dev/null
python api.py &
API_PID=$!
popd  >/dev/null

# wait for port 8000 to be listening (timeout after ~5s)
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

# give it a moment if needed
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
# 5) On exit, kill API and sim (cleanup trap will also fire)
# -----------------------------------------------------------------------------
echo "→ grpo.py exited with code $GRPO_EXIT; shutting down background processes…"
kill $API_PID $SIM_PID || true

exit $GRPO_EXIT
