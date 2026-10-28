"""Generate fictitious records from a fixed RNG; no real applicant data."""
from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
n = 20000
sex = rng.integers(0, 2, n)
x1, x2 = rng.normal(size=(2, n))
latent = 1.2*x1 + 0.6*x2 + rng.normal(0, 1, n)
y = (latent > np.quantile(latent, 0.9)).astype(int)
pd.DataFrame({'x1': x1, 'x2': x2, 'sex': sex, 'label': y}).to_csv(
    Path(__file__).with_name('synthetic.csv'), index=False)
