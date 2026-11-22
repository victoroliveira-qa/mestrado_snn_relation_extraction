"""CLI: python -m src.data.prepare_core --output datasets/raw/core/core_official.csv"""
import argparse
from src.data.core import prepare_core_from_huggingface


def main():
    parser = argparse.ArgumentParser(description='Prepare the official CORE train/test dataset with an audit manifest')
    parser.add_argument('--output', default='datasets/raw/core/core_official.csv')
    parser.add_argument('--repo', default='pborchert/CORE')
    parser.add_argument('--revision', default=None, help='Optional immutable HF commit SHA for reproducibility')
    args = parser.parse_args()
    output, manifest = prepare_core_from_huggingface(args.output, args.repo, args.revision)
    print(f'CORE dataset prepared: {output}\nRows: {manifest["row_count"]}; relations: {manifest["num_relations"]}; splits: {manifest["split_counts"]}')


if __name__ == '__main__':
    main()
