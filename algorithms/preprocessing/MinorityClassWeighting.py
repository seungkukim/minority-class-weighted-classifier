"""ALINJob class-only sample weighting and binary AIF360/MAF adapter.

This is not protected-group reweighing: weights depend only on training labels.
Fairness improvement is an empirical question, not a mathematical guarantee.
"""
import numpy as np
import pandas as pd
from numbers import Real
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _floor(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('floor must be a finite positive number')
    if not np.isfinite(value) or value <= 0:
        raise ValueError('floor must be a finite positive number')
    return float(value)


def _labels(labels):
    y = np.asarray(labels)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not len(y):
        raise ValueError('labels must have shape (n,) or (n, 1), with n > 0')
    if np.any(pd.isna(y)):
        raise ValueError('labels must not contain missing values')
    if np.issubdtype(y.dtype, np.number) and not np.all(np.isfinite(y)):
        raise ValueError('labels must be finite')
    if y.dtype == object and any(isinstance(value, Real) and not np.isfinite(value) for value in y):
        raise ValueError('labels must be finite')
    return y


def _instance_weights(dataset, n):
    weights = np.asarray(dataset.instance_weights, dtype=float)
    if weights.shape != (n,):
        raise ValueError('instance_weights must have shape (n,)')
    with np.errstate(over='ignore'):
        total = weights.sum()
    if not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError('Instance weights must be finite, nonnegative, and have positive total')
    return weights


def minority_weights(labels, floor=250):
    """Class multipliers from raw training counts (not weighted frequencies)."""
    floor, y = _floor(floor), _labels(labels)
    codes, classes = pd.factorize(y, sort=False)
    counts = np.bincount(codes, minlength=len(classes))
    with np.errstate(over='raise'):
        return np.maximum(1.0, floor / counts[codes])


class MinorityClassWeighting:
    """Train-fitted weight transform. Calling transform never learns new counts.

    Apply only to training data for learning; evaluation weights stay unchanged.
    Unseen classes are rejected rather than silently learning from new labels.
    """
    def __init__(self, floor=250):
        self.floor = _floor(floor)

    def fit(self, dataset):
        y = _labels(dataset.labels)
        _instance_weights(dataset, len(y))
        codes, classes = pd.factorize(y, sort=False)
        counts = np.bincount(codes, minlength=len(classes))
        self.class_weights_ = {label: max(1.0, self.floor / count)
                               for label, count in zip(classes, counts)}
        return self

    def transform(self, dataset):
        if not hasattr(self, 'class_weights_'):
            raise RuntimeError('Call fit on the training dataset before transform')
        y = _labels(dataset.labels)
        base = _instance_weights(dataset, len(y))
        if any(label not in self.class_weights_ for label in y):
            raise ValueError('transform received a class absent from fit')
        try:
            with np.errstate(over='raise', invalid='raise'):
                weights = base * np.array([self.class_weights_[label] for label in y])
        except FloatingPointError as error:
            raise ValueError('Transformed instance weights overflowed') from error
        if not np.all(np.isfinite(weights)):
            raise ValueError('Transformed instance weights overflowed')
        result = dataset.copy(deepcopy=True)
        result.instance_weights = weights
        _instance_weights(result, len(y))  # The total can overflow even with finite elements.
        return result

    def fit_transform(self, dataset):
        return self.fit(dataset).transform(dataset)


class ALINJobMinorityWeighting:
    """Guide-compatible binary benchmark adapter; run returns two metric dicts.

    protected=0 denotes the unprivileged group; protected=1 the privileged group.
    The protected feature is removed before either classifier is trained.
    """
    def __init__(self, dataset_name='synthetic', protected='sex', floor=None,
                 seed=1, dataset=None, test_size=0.3, data_path=None):
        self.dataset_name, self.protected = dataset_name, protected
        if floor is None:  # bundled synthetic data is larger, so its floor is scaled up
            floor = 2000 if dataset_name == 'synthetic' and dataset is None else 250
        self.floor, self.seed = _floor(floor), seed
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)) or not 0 <= seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        if not isinstance(test_size, Real) or isinstance(test_size, (bool, np.bool_)) or not 0 < test_size < 1:
            raise ValueError('test_size must be a fraction strictly between 0 and 1')
        self.test_size, self.data_path = float(test_size), data_path
        self.privileged_groups = [{protected: 1}]
        self.unprivileged_groups = [{protected: 0}]
        self.load_and_preprocess_data(dataset)

    def load_and_preprocess_data(self, dataset=None):
        if dataset is None:
            if self.dataset_name == 'synthetic':
                from pathlib import Path
                import pandas as pd
                from aif360.datasets import BinaryLabelDataset
                source = (Path(self.data_path) if self.data_path is not None else
                          Path(__file__).resolve().parents[2] / 'data/synthetic.csv')
                frame = pd.read_csv(source)
                dataset = BinaryLabelDataset(df=frame, label_names=['label'],
                    protected_attribute_names=[self.protected], favorable_label=1,
                    unfavorable_label=0)
            else:
                raise ValueError('Use synthetic or dataset=')
        from aif360.datasets import BinaryLabelDataset
        if not isinstance(dataset, BinaryLabelDataset):
            raise TypeError('dataset must be an AIF360 BinaryLabelDataset')
        dataset = dataset.copy(deepcopy=True)
        y = _labels(dataset.labels)
        if dataset.labels.shape != (len(y), 1) or set(y) != {dataset.favorable_label, dataset.unfavorable_label}:
            raise ValueError('MAF adapter requires both binary classes with matching label metadata')
        if not np.isfinite(dataset.favorable_label) or not np.isfinite(dataset.unfavorable_label):
            raise ValueError('Label metadata must be finite')
        if self.protected not in dataset.protected_attribute_names:
            raise ValueError('Protected attribute missing from dataset metadata')
        if dataset.protected_attributes.shape != (len(y), len(dataset.protected_attribute_names)):
            raise ValueError('protected_attributes shape must match rows and names')
        if not np.all(np.isfinite(dataset.protected_attributes)):
            raise ValueError('Protected attributes must be finite')
        col = dataset.protected_attribute_names.index(self.protected)
        if not np.array_equal(np.unique(dataset.protected_attributes[:, col]), [0, 1]):
            raise ValueError('Protected groups must be encoded 0 and 1')
        weights = _instance_weights(dataset, len(y))
        if dataset.features.ndim != 2 or dataset.features.shape[0] != len(y):
            raise ValueError('features must have shape (n, p)')
        if not np.all(np.isfinite(dataset.features)):
            raise ValueError('Features must be finite; fit imputation on training data before use')
        if len(dataset.feature_names) != dataset.features.shape[1] or len(set(dataset.feature_names)) != len(dataset.feature_names):
            raise ValueError('Feature names must be unique and match feature columns')
        train, test = train_test_split(np.arange(len(y)), test_size=self.test_size,
            random_state=self.seed, stratify=y)
        for indices, partition in [(train, 'train'), (test, 'test')]:
            for label in np.unique(y):
                if weights[indices][y[indices] == label].sum() <= 0:
                    raise ValueError(f'{partition} class {label} has zero effective weight; choose another split or weights')
        self.train_indices_, self.test_indices_ = train.copy(), test.copy()
        self.dataset_orig = dataset
        self.dataset_orig_train = dataset.subset(train)
        self.dataset_orig_test = dataset.subset(test)
        # Remove every protected attribute, including additional ones in custom data.
        self.feature_indices = [i for i, name in enumerate(dataset.feature_names)
                                if name not in dataset.protected_attribute_names]
        if not self.feature_indices:
            raise ValueError('At least one non-protected feature is required')

    def fairness_algorithm_fit(self):
        self.weighting_ = MinorityClassWeighting(self.floor)
        return self.weighting_.fit_transform(self.dataset_orig_train)

    def baseline_fit_predict(self, dataset_train, dataset_test):
        scaler = StandardScaler()
        X = scaler.fit_transform(dataset_train.features[:, self.feature_indices])
        model = LogisticRegression(max_iter=2000, random_state=self.seed)
        # A nonconverged model must not silently become a benchmark result.
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(X, dataset_train.labels.ravel(), sample_weight=dataset_train.instance_weights)
        Xt = scaler.transform(dataset_test.features[:, self.feature_indices])
        pred = dataset_test.copy(deepcopy=True)
        pred.labels = model.predict(Xt).reshape(-1, 1)
        pos = np.flatnonzero(model.classes_ == dataset_train.favorable_label)[0]
        pred.scores = model.predict_proba(Xt)[:, pos].reshape(-1, 1)
        return pred

    def compute_metrics(self, pred):
        from aif360.metrics import ClassificationMetric
        true = self.dataset_orig_test
        metric = ClassificationMetric(true, pred,
            unprivileged_groups=self.unprivileged_groups,
            privileged_groups=self.privileged_groups)
        methods = {
            'true_negative_rate': 'specificity', 'true_positive_rate': 'sensitivity',
            'false_positive_rate': 'false_positive_rate', 'false_negative_rate': 'false_negative_rate',
            'precision': 'precision', 'false_discovery_rate': 'false_discovery_rate',
            'false_omission_rate': 'false_omission_rate', 'accuracy': 'accuracy',
            'error_rate': 'error_rate', 'average_odds_difference': 'average_odds_difference',
            'average_abs_odds_difference': 'average_abs_odds_difference',
            'selection_rate': 'selection_rate', 'disparate_impact': 'disparate_impact',
            'statistical_parity_difference': 'statistical_parity_difference',
            'generalized_entropy_index': 'generalized_entropy_index', 'theil_index': 'theil_index',
            'equal_opportunity_difference': 'equal_opportunity_difference'}
        with np.errstate(divide='ignore', invalid='ignore'):
            result = {key: float(getattr(metric, method)()) for key, method in methods.items()}
        result['Balanced accuracy'] = 0.5 * (result['true_positive_rate'] + result['true_negative_rate'])
        col = true.protected_attribute_names.index(self.protected)
        group = true.protected_attributes[:, col]
        supports = {}
        for value, name in [(0, 'unprivileged'), (1, 'privileged')]:
            mask = (group == value) & (true.labels.ravel() == true.favorable_label)
            supports[name] = float(true.instance_weights[mask].sum())
            result['positive_support_' + name] = int(mask.sum())
            result['positive_weight_' + name] = supports[name]
            result['group_support_' + name] = int((group == value).sum())
            result['tpr_' + name] = (float(metric.true_positive_rate(privileged=bool(value)))
                                     if supports[name] > 0 else None)
        result['deo_measurable'] = all(s > 0 for s in supports.values())
        if not result['deo_measurable']:
            result['equal_opportunity_difference'] = None
        result['absolute_deo'] = (abs(result['equal_opportunity_difference'])
                                  if result['deo_measurable'] else None)
        # Undefined group metrics must not be serialized as NaN or a false zero gap.
        for key, value in result.items():
            if isinstance(value, float) and not np.isfinite(value):
                result[key] = None
        return result

    def run(self):
        orig = self.baseline_fit_predict(self.dataset_orig_train, self.dataset_orig_test)
        weighted = self.fairness_algorithm_fit()
        transf = self.baseline_fit_predict(weighted, self.dataset_orig_test)
        return self.compute_metrics(orig), self.compute_metrics(transf)
