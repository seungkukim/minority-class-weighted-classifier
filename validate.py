"""Offline regression tests: python validate.py (no optimized-away asserts)."""
import unittest
import tempfile
import shutil
import runpy
import zipfile
import hashlib
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from algorithms.preprocessing.MinorityClassWeighting import (
    minority_weights, MinorityClassWeighting, ALINJobMinorityWeighting)


def dataset(n=120):
    frame = pd.DataFrame({'x': np.linspace(-2, 2, n),
                          'sex': np.tile([0, 1], n // 2),
                          'label': np.tile([0, 0, 1, 1], n // 4)})
    return BinaryLabelDataset(df=frame, label_names=['label'],
        protected_attribute_names=['sex'], favorable_label=1, unfavorable_label=0)


class WeightTests(unittest.TestCase):
    def test_multiclass_counts_and_minority_only(self):
        np.testing.assert_allclose(minority_weights(['a', 'a', 'a', 'b', 'c'], 2), [1,1,1,2,2])
        np.testing.assert_array_equal(minority_weights(np.array([[0], [1]]), 1), [1,1])

    def test_matches_original_alinjob_formula(self):
        source = Path(__file__).resolve().parent / 'experiments/scout/train_edu_gbm.py'
        original = runpy.run_path(str(source))['compute_minority_weights']
        labels = ['대졸'] * 800 + ['고졸'] * 40 + ['석사'] * 120 + ['박사'] * 5
        for floor in [1, 150, 250, 500]:
            with self.subTest(floor=floor):
                np.testing.assert_allclose(minority_weights(labels, floor), original(labels, floor))

    def test_invalid_labels_and_floor(self):
        for y in [[], [[0,1],[1,0]], [0, np.nan], [None, 'a'], [0, np.inf]]:
            with self.subTest(y=y), self.assertRaises(ValueError):
                minority_weights(y)
        for value in [0, -1, float('nan'), float('inf'), True, '250', [250]]:
            with self.subTest(floor=value), self.assertRaises(ValueError):
                minority_weights([0,1], value)

    def test_transform_uses_training_counts(self):
        train = dataset()
        train.labels[:] = 0
        train.labels[-20:] = 1
        transform = MinorityClassWeighting(floor=40).fit(train)
        probe = train.subset([0, 119])
        np.testing.assert_allclose(transform.transform(probe).instance_weights, [1,2])
        # Recalculating counts on probe would incorrectly give [40,40].
        with self.assertRaises(RuntimeError):
            MinorityClassWeighting().transform(train)
        probe.labels[0] = 7
        with self.assertRaises(ValueError):
            transform.transform(probe)

    def test_base_weights_and_immutability(self):
        train = dataset()
        train.instance_weights = np.linspace(0, 3, len(train.labels))
        original = train.copy(deepcopy=True)
        result = MinorityClassWeighting(120).fit_transform(train)
        self.assertTrue(train == original)
        np.testing.assert_allclose(result.instance_weights, original.instance_weights * 2)
        np.testing.assert_array_equal(result.features, original.features)
        changed = train.copy(deepcopy=True)
        changed.protected_attributes = 1 - changed.protected_attributes
        np.testing.assert_array_equal(result.instance_weights,
            MinorityClassWeighting(120).fit_transform(changed).instance_weights)

    def test_invalid_base_weights(self):
        for weights in [np.ones((120,1)), np.ones(119), np.zeros(120),
                        np.full(120,-1), np.full(120,np.nan), np.full(120,np.inf), np.full(120,1e308)]:
            d = dataset()
            d.instance_weights = weights
            with self.subTest(shape=weights.shape), self.assertRaises(ValueError):
                MinorityClassWeighting().fit_transform(d)

    def test_weight_overflow_rejected(self):
        d = dataset()
        d.instance_weights = np.full(120, 1e300)
        with self.assertRaises(ValueError):
            MinorityClassWeighting(floor=1e308).fit_transform(d)

    def test_object_infinite_labels_rejected(self):
        for value in [float('inf'), float('-inf')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                minority_weights(np.array([0, value], dtype=object))

    def test_transformed_total_overflow_rejected(self):
        d = dataset()
        d.instance_weights[:] = 1
        with self.assertRaises(ValueError):
            MinorityClassWeighting(floor=1e308).fit_transform(d)


class AdapterTests(unittest.TestCase):
    def test_reject_invalid_input(self):
        for key, value in [('floor',float('nan')), ('seed',-1), ('seed',1.2),
                           ('seed',True), ('test_size',0), ('test_size',True)]:
            with self.subTest(key=key,value=value), self.assertRaises(ValueError):
                ALINJobMinorityWeighting(dataset=dataset(), **{key:value})
        with self.assertRaises(TypeError):
            ALINJobMinorityWeighting(dataset=SimpleNamespace())
        bad = dataset()
        bad.features[0,0] = np.nan
        with self.assertRaises(ValueError):
            ALINJobMinorityWeighting(dataset=bad)
        bad = dataset()
        bad.instance_weights[bad.labels.ravel() == 1] = 0
        with self.assertRaises(ValueError):
            ALINJobMinorityWeighting(dataset=bad)

    def test_multiple_protected_features_excluded(self):
        d = dataset()
        d.feature_names.append('other_group')
        d.features = np.column_stack([d.features, d.protected_attributes[:,0]])
        d.protected_attribute_names.append('other_group')
        d.protected_attributes = np.column_stack([d.protected_attributes, d.protected_attributes[:,0]])
        d.privileged_protected_attributes.append(np.array([1.]))
        d.unprivileged_protected_attributes.append(np.array([0.]))
        runner = ALINJobMinorityWeighting(dataset=d)
        self.assertEqual([runner.dataset_orig.feature_names[i] for i in runner.feature_indices], ['x'])
        self.assertFalse(set(runner.train_indices_) & set(runner.test_indices_))
        d.features[0,0] = 999
        self.assertNotEqual(runner.dataset_orig.features[0,0], 999)

    def test_repeatability(self):
        runner = ALINJobMinorityWeighting()
        before, after = runner.run()
        self.assertEqual((before, after), ALINJobMinorityWeighting().run())
        self.assertTrue(after['deo_measurable'])
        self.assertLessEqual(after['absolute_deo'], 1)
        self.assertEqual(after['positive_support_unprivileged'], 295)

    def test_bundled_synthetic_meets_targets_on_five_seeds(self):
        for seed in range(1, 6):
            before, after = ALINJobMinorityWeighting(seed=seed).run()
            self.assertGreater(after['accuracy'], 0.85, seed)
            self.assertLess(after['absolute_deo'], 0.05, seed)
            self.assertLess(abs(after['accuracy'] - before['accuracy']), 0.03, seed)

    def test_metrics_match_weighted_manual_tpr(self):
        runner = ALINJobMinorityWeighting(dataset=dataset())
        d = runner.dataset_orig_test
        # Evaluation weights are intentionally nonuniform; no training weights leak in.
        d.instance_weights = np.arange(1, len(d.labels)+1, dtype=float)
        pred = d.copy(deepcopy=True)
        pred.labels[::3] = 0
        result = runner.compute_metrics(pred)
        tprs = []
        for group, name in [(0,'unprivileged'), (1,'privileged')]:
            mask = (d.protected_attributes[:,0] == group) & (d.labels.ravel() == 1)
            tpr = np.sum(d.instance_weights[mask] * (pred.labels.ravel()[mask] == 1)) / d.instance_weights[mask].sum()
            self.assertAlmostEqual(result['tpr_' + name], tpr)
            self.assertAlmostEqual(result['positive_weight_' + name], d.instance_weights[mask].sum())
            tprs.append(tpr)
        self.assertAlmostEqual(result['equal_opportunity_difference'], tprs[0]-tprs[1])
        self.assertAlmostEqual(result['absolute_deo'], abs(tprs[0]-tprs[1]))

    def test_missing_or_zero_weight_positive_support(self):
        for zero_weight in [False, True]:
            with self.subTest(zero_weight=zero_weight):
                runner = ALINJobMinorityWeighting(dataset=dataset())
                d = runner.dataset_orig_test
                mask = (d.protected_attributes[:,0] == 0) & (d.labels.ravel() == 1)
                if zero_weight:
                    d.instance_weights[mask] = 0
                else:
                    runner.dataset_orig_test = d.subset(np.flatnonzero(~mask))
                result = runner.compute_metrics(runner.dataset_orig_test.copy(deepcopy=True))
                self.assertIsNone(result['absolute_deo'])
                self.assertIsNone(result['equal_opportunity_difference'])
                self.assertFalse(result['deo_measurable'])

    def test_favorable_label_zero_scores(self):
        d = dataset()
        d.favorable_label, d.unfavorable_label = 0, 1
        runner = ALINJobMinorityWeighting(dataset=d)
        pred = runner.baseline_fit_predict(runner.dataset_orig_train, runner.dataset_orig_test)
        self.assertEqual(pred.scores.shape, pred.labels.shape)
        # Favorable=0 scores must still yield labels with probability threshold 0.5.
        expected = np.where(pred.scores.ravel() > .5, 0, 1)
        np.testing.assert_array_equal(pred.labels.ravel(), expected)
        self.assertTrue(runner.compute_metrics(pred)['deo_measurable'])

    def test_explicit_synthetic_path(self):
        path = Path(__file__).parent / 'data/synthetic.csv'
        before, after = ALINJobMinorityWeighting(data_path=path).run()
        self.assertGreater(before['accuracy'], 0)
        self.assertGreater(after['accuracy'], 0)


class PackagingTests(unittest.TestCase):
    def test_private_data_excluded_and_reproducible(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'package'
            shutil.copytree(source, target, ignore=shutil.ignore_patterns('.git', '.venv', '__pycache__', 'dist', 'outputs', 'downloads', 'private'))
            (target / 'data/private_applicants.csv').write_text('must-not-ship')
            (target / 'REVIEW.md').write_text('internal review')
            (target / 'validation').mkdir(exist_ok=True)
            (target / 'validation/performance.json').write_text('{}')
            module = runpy.run_path(str(target / 'build_submission.py'))
            with contextlib.redirect_stdout(io.StringIO()):
                archive = module['build']()
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                module['build']()
            self.assertEqual(hashlib.sha256(archive.read_bytes()).hexdigest(), digest)
            with zipfile.ZipFile(archive) as output:
                self.assertFalse(any('private_applicants' in name for name in output.namelist()))
                self.assertFalse(any('/validation/' in name or name.endswith('/REVIEW.md') for name in output.namelist()))
                self.assertIsNone(output.testzip())


if __name__ == '__main__':
    unittest.main(verbosity=2)
