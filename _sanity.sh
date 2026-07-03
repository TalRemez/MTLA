#!/bin/bash
# Sanity: all 4 configs at --n 2 --limit 16, full generate->extract->evaluate. Validates the full
# slot x {local,global} extraction end to end. COCO evaluates --agg sum; video default max.
set -u
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate mtla 2>/dev/null
cd /efs/user_folders/talremez/MTLA
export PYTHONUNBUFFERED=1
mkdir -p runs_logs
stamp() { date -u +%H:%M:%S; }

run() {
  local name="$1" cfg="$2" evalflags="$3"
  echo "### $name  START $(stamp)"
  rm -rf "$(python -c "from mtla.config import load_config as L;print(L('$cfg').path('predictions'))")" \
         "$(python -c "from mtla.config import load_config as L;print(L('$cfg').path('features'))")" 2>/dev/null
  echo "  [$name] generate  $(stamp)"
  python -m generate --config "$cfg" --n 2 --limit 16 > "runs_logs/s_$name.generate.log" 2>&1
  echo "  [$name] extract   $(stamp)"
  python -m extract --config "$cfg" > "runs_logs/s_$name.extract.log" 2>&1
  echo "  [$name] evaluate ($evalflags)  $(stamp)"
  python -m evaluate --config "$cfg" $evalflags > "runs_logs/s_$name.evaluate.log" 2>&1
  echo "  [$name] RESULT:"
  grep -vE "Loading|Warning|warn|it/s|INFO|EngineCore|Capturing" "runs_logs/s_$name.evaluate.log" | tail -9
  echo "  [$name] errors: $(grep -icE 'traceback|outofmemory|worker.*failed' runs_logs/s_$name.*.log)"
  echo "### $name DONE $(stamp)"
}

echo "===== SANITY START $(date -u) ====="
run coco_internvl    configs/coco_internvl.yaml        "--agg sum"
run coco_qwen3vl     configs/coco_qwen3vl.yaml         "--agg sum"
run qvh_qwen3vl      configs/qvhighlights_qwen3vl.yaml ""
run charades_qwen3vl configs/charades_qwen3vl.yaml     ""
echo "===== SANITY DONE $(date -u) ====="
