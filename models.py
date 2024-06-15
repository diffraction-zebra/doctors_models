from typing import Optional, Dict
import pathlib

import pandas as pd
from sktime.utils import mlflow_sktime

forecasters: Optional[Dict] = None


def load_models():
    global forecasters
    forecasters = {}
    models_path = pathlib.Path('weights')
    assert models_path.exists()
    for model_dir in models_path.iterdir():
        forecaster = mlflow_sktime.load_model(str(model_dir))
        forecasters[model_dir.name] = forecaster


def predict(fh: pd.PeriodIndex) -> pd.DataFrame:
    assert forecasters is not None
    assert fh.dtype == 'period[W-SUN]'
    df = pd.DataFrame(data=[], index=fh)
    for survey, f in forecasters.items():
        df[survey] = f.predict(fh)
    return df


def update(true_values: pd.DataFrame) -> None:
    assert forecasters is not None
    assert set(true_values.columns) == set(forecasters.keys())
    assert true_values.index.dtype == 'period[W-SUN]'

    for survey, forecaster in forecasters.items():
        forecaster.update(true_values[survey], update_params=False)
