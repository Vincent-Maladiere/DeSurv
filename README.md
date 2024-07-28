## Getting started

Create a virtual environment and activate it:
```
python -m venv desurv_env
source desurv_env/bin/activate
```

Install dependencies:
```
python -m pip install -r requirements.txt
```

## Run the benchmark

To run the benchmark for DeSurv on SEER (or the synthetic dataset using a weibull distribution):

```python
from evaluate import run_evaluation

run_evaluation("seer") # or "weibull"
```

