#!/usr/bin/env python3

import sys
from pathlib import Path

import pyvista as pv


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: vtk2ply.py input.vtk [output.ply]")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    output_path = (
        Path(sys.argv[2])
        if len(sys.argv) == 3
        else input_path.with_suffix(".ply")
    )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Input VTK file does not exist: {input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read the original VTK object.
    mesh = pv.read(str(input_path))

    # Convert volumetric or other VTK datasets to an explicit surface.
    # For PolyData, extract_surface preserves the polygonal surface.
    mesh = mesh.extract_surface()

    # The evaluator requires triangular faces. Triangulation preserves the
    # direction of the original polygon winding but splits non-triangular
    # polygons into triangles.
    mesh = mesh.triangulate()

    if mesh.n_points == 0:
        raise ValueError(
            f"Converted mesh contains no points: {input_path}"
        )

    if mesh.n_cells == 0:
        raise ValueError(
            f"Converted mesh contains no cells: {input_path}"
        )

    # Do not use:
    #
    #     auto_orient_normals=True
    #     consistent_normals=True
    #
    # Those options can repair or change face winding, which would invalidate
    # the orientation-consistency quality metric. PLY files do not require
    # precomputed normals for this evaluation.
    mesh.save(
        str(output_path),
        binary=False,
    )

    if not output_path.is_file():
        raise RuntimeError(
            f"PLY output was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise RuntimeError(
            f"PLY output is empty: {output_path}"
        )

    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()