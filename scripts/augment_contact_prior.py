#!/usr/bin/env python3
"""Create a separate binary prior union with audited semantic contact candidates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.semantic_prior import augment_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--semantic-cues', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(augment_archive(args.input, args.semantic_cues, args.output)))


if __name__ == '__main__':
    main()
