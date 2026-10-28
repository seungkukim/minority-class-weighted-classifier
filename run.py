"""MAF-style binary benchmark. JSON stdout contains no progress messages."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
from algorithms.preprocessing.MinorityClassWeighting import ALINJobMinorityWeighting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['synthetic'], default='synthetic')
    parser.add_argument('--protected', default='sex')
    parser.add_argument('--floor', type=float, help='default: 2000 for synthetic, 250 otherwise')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--test-size', type=float, default=0.3)
    parser.add_argument('--data-path', type=Path, help='Synthetic CSV location after copying the algorithm into MAF')
    parser.add_argument('--output', type=Path, help='Also save the JSON report to this file')
    args = parser.parse_args()
    if args.data_path is not None and args.dataset != 'synthetic':
        parser.error('--data-path is supported only with --dataset synthetic')
    # Some upstream loaders write notices to stdout. Keep the report parseable.
    try:
        with contextlib.redirect_stdout(sys.stderr):
            runner = ALINJobMinorityWeighting(args.dataset, args.protected, args.floor, args.seed,
                test_size=args.test_size, data_path=args.data_path)
            before, after = runner.run()
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    payload = {'dataset': args.dataset, 'seed': args.seed, 'floor': runner.floor,
        'test_size': args.test_size, 'n_train': len(runner.train_indices_),
        'n_test': len(runner.test_indices_),
        'baseline': before, 'weighted': after,
        'accuracy_gap_pp': 100 * abs(after['accuracy'] - before['accuracy'])}
    report = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output is not None:
        args.output.write_text(report, encoding='utf-8')
    print(report, end='')


if __name__ == '__main__':
    main()
