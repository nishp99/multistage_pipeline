"""Shared command-line and result handling for the paper experiments."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run(description, name, defaults, quick_defaults, compute, plot):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced dimensions, grids and durations.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / name)
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON overrides of the settings listed by --show-config.",
    )
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw the most recent completed result.",
    )
    args = parser.parse_args()
    cfg = defaults | (quick_defaults if args.quick else {})
    if args.config:
        overrides = json.loads(args.config.read_text())
        unknown = overrides.keys() - cfg.keys()
        if unknown:
            parser.error(f"Unknown settings: {sorted(unknown)}")
        cfg.update(overrides)
    if args.show_config:
        print(json.dumps(cfg, indent=2))
        return
    cfg["quick"] = args.quick
    out = args.output_dir
    out.mkdir(exist_ok=True, parents=True)
    path = out / "result.npz"
    if not args.plot_only:
        arrays = compute(cfg)
        with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as temp:
            np.savez_compressed(temp, **arrays, config_json=json.dumps(cfg))
        os.replace(temp.name, path)
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        fig = plot(data, config)
        fig.savefig(out / "figure.pdf", bbox_inches="tight")
        plt.close(fig)
    print(f"Results: {path}\nFigure: {out/'figure.pdf'}")
