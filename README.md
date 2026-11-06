# One-hidden-layer neural network from scratch (MAT0611)

Individual course project for MAT0611 "Mathematics for Machine Learning" (Albert School).

Implementation, from scratch with NumPy, of a neural network with one hidden layer:
- `SimpleSLPClassifier` (binary and multi-class classification)
- `SimpleSLPRegressor` (regression)

Both follow the scikit-learn interface (`fit`, `predict`, `score`).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Files
| File | Role |
|---|---|
| `activations.py` | Activation functions and their derivatives |
| `slp_base.py` | Shared base class |
| `slp_regressor.py` | Regressor |
| `slp_classifier.py` | Classifier |
| `test_*.py` | Tests provided with the project |
| `neural_network_project.pdf` | Project statement |
