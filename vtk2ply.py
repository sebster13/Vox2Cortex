#!/usr/bin/env python3
import sys
import pyvista as pv
from pathlib import Path

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Usage: vtk2ply.py input.vtk [output.ply]")
    sys.exit(1)

inp = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) == 3 else inp.with_suffix(".ply")

# Read
mesh = pv.read(str(inp))

# Ensure a surface polydata and triangles (PLY expects polygonal, best as triangles)
mesh = mesh.extract_surface().triangulate()

# Optional: make sure normals exist for better shading
if "Normals" not in mesh.point_data:
    mesh = mesh.compute_normals(auto_orient_normals=True, consistent_normals=True)

# Write (binary=True gives smaller files; set False for human-readable)
mesh.save(str(out), binary=False)
print(f"Wrote: {out}")
