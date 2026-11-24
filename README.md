# Python-Programming-for-Machine-Learning

Scenario: You are a junior ML engineer at NorthWave Telecom. The analytics team has exported a customer dataset. It is messy: it has missing values, numbers stored as text, inconsistent category labels, data-entry outliers and duplicate rows. The retention team wants a churn model that flags at least 75% of customers who are about to leave. They also want a small HTTP service so that the CRM can request predictions.

You will build a small, tested Python package called mlkit. It contains NumPy utilities, a pandas cleaning module, headless visualisations, scikit-learn pipelines, hyperparameter search, a tuned decision threshold, a persisted model artefact, a batch prediction CLI and a FastAPI service. You will exercise the service with curl.

What you will build:

~/churn-lab/
  pyproject.toml, requirements.txt, requirements.lock
  scripts/      generate_data.py, vectorize_demo.py, eda.py, plots.py, broken_train.py
  mlkit/        numops.py, data_io.py, clean.py, features.py, train.py, tune.py, predict.py, api.py
  tests/        test_numops.py, test_clean.py, test_pipeline.py, test_threshold.py
  data/         raw/, processed/, new_customers.csv, api_request.json
  reports/      figures/*.png, metrics.json, predictions.csv
  models/       churn_model.joblib, model_metadata.json
  validate.sh
