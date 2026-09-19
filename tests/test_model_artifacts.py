import pickle
import numpy as np
import pandas as pd
import pytest
from src.artifacts import validate_predictions
from src.model import FittedRiskModel, _pipeline, temporal_train_mask

def test_temporal_boundary_purges_overlapping_windows():
    d = pd.DataFrame({"forecast_origin_year": [2008, 2009, 2010], "label_available_year": [2011, 2012, 2013]})
    assert temporal_train_mask(d, 2013).tolist() == [True, True, False]

def test_transforms_fit_and_artifact_round_trip():
    x = pd.DataFrame({"x": [0., np.nan, 2., 4.]}); y = [0, 0, 1, 1]
    fitted = FittedRiskModel(_pipeline(["x"], .1).fit(x, y), None, ["x"])
    restored = pickle.loads(pickle.dumps(fitted))
    assert np.allclose(fitted.predict(x), restored.predict(x))

def test_prediction_schema_null_risk_for_unavailable():
    from src.artifacts import REQUIRED
    row = {c: "x" for c in REQUIRED}; row.update(model_type="democratic_breakdown", estimated_risk=np.nan,
        score_status="unavailable", unavailable_reason="not_democratic_at_origin")
    validate_predictions(pd.DataFrame([row]))
    row["estimated_risk"] = 0.0
    with pytest.raises(ValueError, match="null risk"): validate_predictions(pd.DataFrame([row]))

def test_join_missing_inputs_remain_missing():
    left = pd.DataFrame({"country_id": ["AAA", "BBB"], "year": [2000, 2000]})
    right = pd.DataFrame({"country_id": ["AAA"], "year": [2000], "x": [1]})
    joined = left.merge(right, how="left", on=["country_id", "year"], validate="one_to_one")
    assert pd.isna(joined.loc[joined.country_id.eq("BBB"), "x"]).item()

def test_scoring_does_not_require_future_labels():
    train = pd.DataFrame({"x": [0., 1., 2., 3.], "outcome": [0, 0, 1, 1]})
    model = FittedRiskModel(_pipeline(["x"]).fit(train[["x"]], train.outcome), None, ["x"])
    latest = pd.DataFrame({"x": [1.5]})
    assert 0 <= model.predict(latest)[0] <= 1
