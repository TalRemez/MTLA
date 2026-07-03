#!/bin/bash
# Overnight: (1) 100-example N=16 sanity on all 4 configs; (2) if all 4 produced valid results,
# the FULL datasets N=16. Each config: fresh generate -> extract -> evaluate. COCO evaluates
# --agg sum (headline mAP); AUROC is agg-independent. Video default max.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mtla 2>/dev/null
cd /efs/user_folders/talremez/MTLA
export PYTHONUNBUFFERED=1
mkdir -p runs_logs
stamp() { date -u +%H:%M:%S; }

# name  config  evalflags
CONFIGS=(
  "coco_internvl|configs/coco_internvl.yaml|--agg sum"
  "coco_qwen3vl|configs/coco_qwen3vl.yaml|--agg sum"
  "qvh_qwen3vl|configs/qvhighlights_qwen3vl.yaml|"
  "charades_qwen3vl|configs/charades_qwen3vl.yaml|"
)

# run <tag> <name> <cfg> <evalflags> <limit-or-empty>; returns 0 if evaluate produced results.
run() {
  local tag="$1" name="$2" cfg="$3" evalflags="$4" limit="$5"
  local lim_arg=""; [ -n "$limit" ] && lim_arg="--limit $limit"
  echo "### [$tag] $name  START $(stamp)  (n=16 $lim_arg)"
  rm -rf "$(python -c "from mtla.config import load_config as L;print(L('$cfg').path('predictions'))")" \
         "$(python -c "from mtla.config import load_config as L;print(L('$cfg').path('features'))")" 2>/dev/null
  python -m generate --config "$cfg" --n 16 $lim_arg > "runs_logs/${tag}_$name.generate.log" 2>&1
  python -m extract  --config "$cfg"              > "runs_logs/${tag}_$name.extract.log" 2>&1
  python -m evaluate --config "$cfg" $evalflags   > "runs_logs/${tag}_$name.evaluate.log" 2>&1
  echo "  [$tag/$name] RESULT:"
  grep -vE "Loading|Warning|warn|it/s|INFO|EngineCore|Capturing" "runs_logs/${tag}_$name.evaluate.log" | tail -9
  local errs; errs=$(grep -licE 'traceback|outofmemory|worker.*failed' runs_logs/${tag}_$name.*.log | grep -v ':0$' | wc -l)
  echo "  [$tag/$name] error-logs=$errs  DONE $(stamp)"
  grep -q 'auroc_mtla' "runs_logs/${tag}_$name.evaluate.log"  # success = produced an AUROC
}

echo "===== OVERNIGHT START $(date -u) ====="

echo "########## PHASE 1: 100-example N=16 sanity ##########"
ok=1
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name cfg ev <<< "$c"
  run s100 "$name" "$cfg" "$ev" 100 || { echo "!!! [$name] PHASE1 FAILED"; ok=0; }
done

if [ "$ok" != "1" ]; then
  echo "===== PHASE 1 had a failure; NOT running full sets. $(date -u) ====="
  exit 1
fi

echo "########## PHASE 2: FULL datasets N=16 ##########"
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name cfg ev <<< "$c"
  run full "$name" "$cfg" "$ev" "" || echo "!!! [$name] PHASE2 FAILED (continuing)"
done

echo "===== OVERNIGHT DONE $(date -u) ====="
