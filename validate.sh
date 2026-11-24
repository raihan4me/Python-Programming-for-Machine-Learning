#!/usr/bin/env bash
# Final validation for the churn lab
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
PASS=0; FAIL=0
check() {
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then echo "[PASS] $desc"; PASS=$((PASS+1))
  else echo "[FAIL] $desc"; FAIL=$((FAIL+1)); fi
}

check "Task 1: venv has scikit-learn 1.5.1 and pandas 2.2.2" \
  $PY -c "import sys, sklearn, pandas; sys.exit(not (sklearn.__version__=='1.5.1' and pandas.__version__=='2.2.2'))"
check "Task 1: mlkit importable (editable install)" $PY -c "import mlkit"
check "Task 1: requirements.lock exists" test -s requirements.lock
check "Task 1: raw dataset has 5050 rows" \
  $PY -c "import sys; from mlkit.data_io import read_customers as r; sys.exit(len(r('data/raw/customers.csv'))!=5050)"
check "Task 2: NumPy tests pass" $PY -m pytest -q tests/test_numops.py
check "Task 3: EDA script runs" $PY scripts/eda.py
check "Task 4: cleaning tests pass" $PY -m pytest -q tests/test_clean.py
check "Task 4: processed data = 5000 rows, 3 contracts, 0/1 target, no NaN in internet_service" \
  $PY -c "
import sys
from mlkit.data_io import read_customers
d = read_customers('data/processed/customers_clean.csv')
ok = len(d)==5000 and d['contract'].nunique()==3 and set(d['churn'].unique())=={0,1} \
     and d['internet_service'].isna().sum()==0 and d['monthly_charges'].max()<=200
sys.exit(not ok)"
check "Task 5: four 1000x600 PNG figures" \
  bash -c 'test "$(file reports/figures/*.png | grep -c "PNG image data, 1000 x 600")" -ge 4'
check "Task 6: pipeline tests pass" $PY -m pytest -q tests/test_pipeline.py
check "Task 6: metrics.json has 2 models with test ROC AUC >= 0.70" \
  $PY -c "
import json, sys
m = json.load(open('reports/metrics.json'))
sys.exit(not (len(m)==2 and all(v['test']['roc_auc']>=0.70 for v in m.values())))"
check "Task 7: threshold test passes" $PY -m pytest -q tests/test_threshold.py
check "Task 7: metadata records version, threshold and tuned recall >= 0.65" \
  $PY -c "
import json, sys
m = json.load(open('models/model_metadata.json'))
sys.exit(not (m['sklearn']=='1.5.1' and 0<m['threshold']<1 and m['test_metrics_tuned_threshold']['recall']>=0.65))"
check "Task 7: persisted model loads and predicts" \
  $PY -c "
import joblib
from mlkit.train import load_xy
X, _ = load_xy(); joblib.load('models/churn_model.joblib').predict_proba(X.head(5))"
check "Task 8: predictions.csv has 5 rows" \
  $PY -c "import sys, pandas as pd; sys.exit(len(pd.read_csv('reports/predictions.csv'))!=5)"

VAL_PID=""
if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  .venv/bin/uvicorn mlkit.api:app --host 127.0.0.1 --port 8000 >/tmp/validate_uvicorn.log 2>&1 &
  VAL_PID=$!; sleep 5
fi
check "Task 8: API /health returns ok" bash -c 'curl -sf http://127.0.0.1:8000/health | jq -e ".status==\"ok\""'
check "Task 8: API /predict returns 2 predictions" \
  bash -c 'curl -sf -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @data/api_request.json | jq -e ".predictions | length == 2"'
check "Task 8: API rejects invalid payload with 422" \
  bash -c 'test "$(curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "[{\"customer_id\":\"x\"}]")" = 422'
[ -n "$VAL_PID" ] && kill "$VAL_PID"

echo
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ]
