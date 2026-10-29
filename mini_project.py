import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso

data = load_diabetes()
X, y = data.data, data.target
features = data.feature_names

for alpha in [0.1, 1.0, 10.0]:
    lin = LinearRegression().fit(X, y)
    ridge = Ridge(alpha=alpha).fit(X, y)
    lasso = Lasso(alpha=alpha).fit(X, y)
    zero_features = [features[i] for i, c in enumerate(lasso.coef_) if abs(c) < 1e-6]
    print(f"Alpha={alpha}")
    print("Linear:", np.round(lin.coef_, 2))
    print("Ridge:", np.round(ridge.coef_, 2))
    print("Lasso:", np.round(lasso.coef_, 2))
    print("Lasso zeroed:", zero_features)
    print("---")
