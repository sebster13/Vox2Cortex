import argparse
import meshio
import numpy as np
import sys

def create_distance_heatmap(gt_mesh_path, pred_mesh_path, output_path):
    """
    Calculates the vertex-wise Euclidean distance between a ground truth and a 
    predicted mesh and saves the result with the distances as scalar data.

    Args:
        gt_mesh_path (str): Path to the ground truth (original) mesh file.
        pred_mesh_path (str): Path to the predicted (reconstructed) mesh file.
        output_path (str): Path to save the output mesh with distance scalars.
    """
    # --- 1. Load the meshes ---
    try:
        print(f"Loading ground truth mesh: {gt_mesh_path}")
        gt_mesh = meshio.read(gt_mesh_path)
        
        print(f"Loading predicted mesh: {pred_mesh_path}")
        pred_mesh = meshio.read(pred_mesh_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.", file=sys.stderr)
        sys.exit(1)

    # --- 2. Validate vertex correspondence ---
    if gt_mesh.points.shape != pred_mesh.points.shape:
        print("Error: The number of vertices in the meshes do not match!", file=sys.stderr)
        print(f"Ground truth mesh has {len(gt_mesh.points)} vertices.", file=sys.stderr)
        print(f"Predicted mesh has {len(pred_mesh.points)} vertices.", file=sys.stderr)
        sys.exit(1)
    
    print("Meshes have a matching number of vertices. Proceeding...")

    # --- 3. Calculate Euclidean distance ---
    # Calculate the L2 norm (Euclidean distance) of the difference vector for each vertex.
    distances = np.linalg.norm(pred_mesh.points - gt_mesh.points, axis=1)

    # Print summary statistics to the console
    print("\n--- Distance Statistics ---")
    print(f"Minimum distance: {np.min(distances):.4f}")
    print(f"Maximum distance: {np.max(distances):.4f}")
    print(f"Mean distance:    {np.mean(distances):.4f}")
    print(f"Standard dev:     {np.std(distances):.4f}")

    # --- 4. Prepare mesh for saving ---
    # We use the predicted mesh's geometry and attach the calculated distances.
    # The string 'distance_error' will be the name of our scalar array.
    pred_mesh.point_data['distance_error'] = distances

    # --- 5. Save the output mesh ---
    try:
        print(f"\nSaving mesh with distance data to: {output_path}")
        # The file format is inferred from the output file extension.
        # .vtk is recommended for ParaView/3D Slicer.
        meshio.write(output_path, pred_mesh)
        print("Successfully created heatmap file.")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate vertex-wise distance between two corresponding meshes and save the result as a VTK/PLY file with scalar data for heatmap visualization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--pred',
        type=str,
        required=True,
        help="Path to the predicted/reconstructed surface mesh file."
    )
    parser.add_argument(
        '--gt',
        type=str,
        required=True,
        help="Path to the ground truth/original surface mesh file."
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help="Path for the output file with distance data (.vtk recommended)."
    )

    args = parser.parse_args()

    create_distance_heatmap(
        gt_mesh_path=args.gt,
        pred_mesh_path=args.pred,
        output_path=args.output
    )