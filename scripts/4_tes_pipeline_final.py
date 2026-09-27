"""Generate a review-only dashboard using the repository's shared pipeline."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from all_in_one_yolocomvis import run_dashboard  # noqa: E402


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=ROOT / "ujicoba.png")
    parser.add_argument("--output", type=Path, default=ROOT / "hasil_dashboard_dynamic_attention.jpg")
    args = parser.parse_args()
    run_dashboard(ROOT, args.image.expanduser().resolve(), args.output.expanduser().resolve())
