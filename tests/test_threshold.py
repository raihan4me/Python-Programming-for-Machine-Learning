import numpy as np
import pytest

from mlkit.tune import choose_threshold


def test_choose_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    assert choose_threshold(y, scores, 1.0) == pytest.approx(0.35)
    assert choose_threshold(y, scores, 0.5) == pytest.approx(0.8)
    assert isinstance(choose_threshold(y, scores, 0.5), float)
