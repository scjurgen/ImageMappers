#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple


def make_heightfield(
        size: int = 800,
        x_min: float = -2.0,
        x_max: float = 2.0,
        y_min: float = -2.0,
        y_max: float = 2.0,
        pattern: str = "radial_sine",
        ratio_x: float = 1.0,
        ratio_y: float = 1.0,
        depth: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(x_min, x_max, size)
    y = np.linspace(y_min, y_max, size)
    X, Y = np.meshgrid(x, y)

    if pattern == "radial_sine":
        r = np.sqrt(X ** 2 + Y ** 2)
        theta = np.arctan2(Y, X)
        amp = np.exp(-2.0 * (r - 1.0) ** 2)
        phase = ratio_x * theta
        Z1 = amp * np.sin(phase)

        if ratio_y != 1.0:
            phase2 = ratio_y * theta
            Z2 = amp * np.sin(phase2)
            Z = Z1 + depth * Z2
        else:
            Z = depth * Z1

    elif pattern == "rings":
        r = np.sqrt(X ** 2 + Y ** 2)
        amp = np.exp(-0.5 * r ** 2)
        phase = ratio_x * 4.0 * r
        Z = depth * amp * np.sin(phase)

    elif pattern == "checker":
        k_x = 3.0 * ratio_x
        k_y = 3.0 * ratio_y
        Z = depth * np.sin(k_x * X) * np.sin(k_y * Y)

    elif pattern == "parametric":
        X_mapped = np.sin(ratio_x * X)
        Y_mapped = np.cos(ratio_y * Y)
        Z = depth * (0.5 * X_mapped + 0.5 * Y_mapped)

    else:
        Z = np.zeros_like(X)

    img = 0.5 - 0.5 * Z
    return img, X, Y, Z


def save_2d_map(
        filename: str,
        size: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        pattern: str,
        ratio_x: float,
        ratio_y: float,
        depth: float,
        cmap: str
) -> None:
    img, _, _, _ = make_heightfield(size, x_min, x_max, y_min, y_max, pattern, ratio_x, ratio_y, depth)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.axis("off")
    plt.tight_layout(pad=0.0)
    plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def save_stereo_surface(
        filename: str,
        size: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        pattern: str,
        elev: float,
        azim_center: float,
        eye_sep: float,
        ratio_x: float,
        ratio_y: float,
        depth: float,
        parallel: bool,
        cmap: str
) -> None:
    _, X, Y, Z = make_heightfield(size, x_min, x_max, y_min, y_max, pattern, ratio_x, ratio_y, depth)

    fig = plt.figure(figsize=(10, 4))

    if parallel:
        ax_l = fig.add_subplot(1, 2, 1, projection="3d")
        ax_l.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.0, antialiased=True)
        ax_l.view_init(elev=elev, azim=azim_center - eye_sep)
        ax_l.axis("off")

        ax_r = fig.add_subplot(1, 2, 2, projection="3d")
        ax_r.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.0, antialiased=True)
        ax_r.view_init(elev=elev, azim=azim_center + eye_sep)
        ax_r.axis("off")

    else:
        ax_r = fig.add_subplot(1, 2, 1, projection="3d")
        ax_r.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.0, antialiased=True)
        ax_r.view_init(elev=elev, azim=azim_center + eye_sep)
        ax_r.axis("off")

        ax_l = fig.add_subplot(1, 2, 2, projection="3d")
        ax_l.plot_surface(X, Y, Z, cmap=cmap, linewidth=0.0, antialiased=True)
        ax_l.view_init(elev=elev, azim=azim_center - eye_sep)
        ax_l.axis("off")

    plt.tight_layout(pad=0.0)
    plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def list_colormaps() -> None:
    print("Available colormaps:")
    print("  viridis, magma, plasma, inferno, cividis")
    print("  gray, binary")
    print("  rainbow, jet, turbo")
    print("  twilight, hsv")
    print("  coolwarm, bwr, seismic")
    print("  Spectral, RdYlBu, RdYlGn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 2D gray maps and stereoscopic 3D views."
    )

    parser.add_argument("--size", type=int, default=800,
                        help="image resolution (pixels, default: %(default)s)")
    parser.add_argument("--x-min", type=float, default=-2.0,
                        help="min x (or p for parametric, default: %(default)s)")
    parser.add_argument("--x-max", type=float, default=2.0,
                        help="max x (or p for parametric, default: %(default)s)")
    parser.add_argument("--y-min", type=float, default=-2.0,
                        help="min y (or q for parametric, default: %(default)s)")
    parser.add_argument("--y-max", type=float, default=2.0,
                        help="max y (or q for parametric, default: %(default)s)")
    parser.add_argument("--pattern", type=str, default="radial_sine",
                        choices=["radial_sine", "rings", "checker", "parametric"],
                        help="heightfield pattern (default: %(default)s)")
    parser.add_argument("--ratio-x", type=float, default=1.0,
                        help="frequency ratio for x/p axis (default: %(default)s)")
    parser.add_argument("--ratio-y", type=float, default=1.0,
                        help="frequency ratio for y/q axis (default: %(default)s)")
    parser.add_argument("--depth", type=float, default=1.0,
                        help="depth multiplier (default: %(default)s)")
    parser.add_argument("--cmap", type=str, default="gray",
                        help="colormap (e.g., viridis, magma, rainbow, gray; default: %(default)s)")
    parser.add_argument("--list-cmaps", action="store_true",
                        help="list available colormaps and exit")
    parser.add_argument("--out-2d", type=str, default="map_2d.png",
                        help="output filename for 2D map (default: %(default)s)")
    parser.add_argument("--out-3d", type=str, default="stereo_3d.png",
                        help="output filename for stereo 3D (default: %(default)s)")
    parser.add_argument("--no-2d", action="store_true",
                        help="skip 2D output")
    parser.add_argument("--no-3d", action="store_true",
                        help="skip stereo 3D output")
    parser.add_argument("--elev", type=float, default=40.0,
                        help="3D elevation angle (deg, default: %(default)s)")
    parser.add_argument("--azim", type=float, default=45.0,
                        help="3D center azimuth angle (deg, default: %(default)s)")
    parser.add_argument("--eye-sep", type=float, default=5.0,
                        help="stereo eye separation in azimuth degrees "
                             "(default: %(default)s)")
    parser.add_argument("--parallel", action="store_true",
                        help="use parallel (wall-eyed) viewing instead of cross-eyed")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_cmaps:
        list_colormaps()
        return

    if not args.no_2d:
        save_2d_map(args.out_2d,
                    args.size, args.x_min, args.x_max, args.y_min, args.y_max,
                    args.pattern, args.ratio_x, args.ratio_y, args.depth, args.cmap)

    if not args.no_3d:
        save_stereo_surface(args.out_3d,
                            args.size, args.x_min, args.x_max, args.y_min, args.y_max,
                            args.pattern, args.elev, args.azim, args.eye_sep,
                            args.ratio_x, args.ratio_y, args.depth, args.parallel, args.cmap)


if __name__ == "__main__":
    main()
