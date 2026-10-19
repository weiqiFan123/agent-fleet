#!/usr/bin/env bash
set -euo pipefail
W="${TMAX_WORKSPACE_ROOT:-/mnt/shared-storage-user/weiqi-gpfs/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${HARBOR_TASK_ID:?}"
[[ "$TASK_ID" =~ ^task_[0-9]{6}_[0-9a-f]{8}$ ]] || exit 2
TASK_DIR="${DATASET_PATH:?}/$TASK_ID"
test -s "$TASK_DIR/task.toml"
test -s "$TASK_DIR/instruction.md"
test -s "$TASK_DIR/tests/test.sh"
test -s "$TASK_DIR/tests/test_final_state.py"
# Match Harbor's outer start deadline to the OpenSandbox readiness deadline.
# Scale each task's own build timeout without modifying the pinned task data.
START_TIMEOUT_SEC=3600
BUILD_TIMEOUT_MULTIPLIER="$(python3 - "$TASK_DIR/task.toml" "$START_TIMEOUT_SEC" <<'PY'
import math, sys, tomllib
with open(sys.argv[1], 'rb') as stream:
    config = tomllib.load(stream)
base = float(config.get('environment', {}).get('build_timeout_sec', 600.0))
if not math.isfinite(base) or base <= 0:
    raise SystemExit('task environment.build_timeout_sec must be finite and positive')
print(float(sys.argv[2]) / base)
PY
)"
IMAGE_CACHE="${THREAD_ROOT:?}/manifests/task-images"
mkdir -p "$IMAGE_CACHE"
# One registry lookup per task even with 32 samples starting together.
exec 8>"$IMAGE_CACHE/$TASK_ID.lock"
flock 8
if [[ ! -s "$IMAGE_CACHE/$TASK_ID.json" ]]; then
  # The migrated CPU need not have ripgrep; TASK_ID is validated above and
  # this anchored basic-regexp match has the same task-ref semantics.
  grep "^tmax/${TASK_ID}@sha256:" "${TMAX_TASK_REFS:-/mnt/shared-storage-user/weiqi-gpfs/data/tmax/task_refs.txt}" > "$IMAGE_CACHE/$TASK_ID.ref"
  python3 "${TMAX_IMAGE_RESOLVER:-$W/tmax-phases/scripts/resolve_tmax_harbor_images.py}" \
    "$IMAGE_CACHE/$TASK_ID.ref" "$IMAGE_CACHE/$TASK_ID.json"
fi
IMAGE_REF="$(python3 - "$IMAGE_CACHE/$TASK_ID.json" "$TASK_ID" <<'PY'
import json, re, sys
data = json.load(open(sys.argv[1]))
tags = data['image_results'][sys.argv[2]]['main_tags']
assert len(tags) == 1, 'task must resolve to exactly one main image'
digest = next(iter(tags.values()))
assert re.fullmatch(r'sha256:[0-9a-f]{64}', digest), 'invalid image digest'
print(f'harbor-sandbox.yicloud.com.cn/agent-fleet-tmax/{sys.argv[2]}@{digest}')
PY
)"
flock -u 8
export PYTHONPATH="$SCRIPT_DIR:${TMAX_OFFICIAL_ROOT:-$W/tmax-official}${PYTHONPATH:+:$PYTHONPATH}"
export OPENAI_API_KEY="${API_KEY:-local-no-auth}"
unset YICLOUD_SANDBOX_INSTANCE_ID YICLOUD_SANDBOX_PING_URL
export HARBOR_OPENSANDBOX_BUNDLE_MANIFEST=""
echo "agent=Vanillux2Agent task=$TASK_ID image=$IMAGE_REF"
echo "environment_start_timeout_sec=$START_TIMEOUT_SEC environment_build_timeout_multiplier=$BUILD_TIMEOUT_MULTIPLIER"
cmd=("${HARBOR_CLI_BIN:?}" run --path "$TASK_DIR"
  --env yicloud_opensandbox:YiCloudOpenSandboxEnvironment --ek "image_ref=$IMAGE_REF"
  --ek "ready_timeout_sec=$START_TIMEOUT_SEC"
  --environment-build-timeout-multiplier "$BUILD_TIMEOUT_MULTIPLIER"
  --agent-import-path tmax_vanillux_agent:Vanillux2Agent
  --model "openai/${RL_MODEL_NAME#openai/}"
  --ak "api_base=${RL_API_BASE:?}"
  --ak "temperature=${RL_TEMPERATURE:-1.0}" --ak "top_p=${RL_TOP_P:-0.95}"
  --ak max_steps=64 --ak max_tokens=16384 --ak max_format_errors=64
  --n-concurrent 1 --n-attempts 1 --max-retries 0
  --agent-timeout-multiplier "${RL_AGENT_TIMEOUT_MULTIPLIER:-2.0}"
  --job-name "vanillux-${TASK_ID}-$$" --jobs-dir "${JOBS_ROOT:?}" --yes)
if [[ "${HARBOR_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"; printf '\n'
else
  exec "${cmd[@]}"
fi
