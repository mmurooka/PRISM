#!/usr/bin/env python3
"""Infer contact sequences from aligned human meshes, box poses and visual priors."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prism_reconstruction.contacts import ContactConfig
from prism_reconstruction.pipeline import run_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--friction', type=float, default=.6)
    parser.add_argument('--mass-kg', type=float, default=None, help='Omit to report forces per unit object mass')
    parser.add_argument('--min-region-vertices', type=int, default=3)
    args = parser.parse_args()
    result = run_archive(args.input, args.output, friction=args.friction, mass_kg=args.mass_kg,
                         config=ContactConfig(min_region_vertices=args.min_region_vertices))
    print(f"Saved {args.output}; force unit: {result['force_unit']}")


if __name__ == '__main__':
    main()
