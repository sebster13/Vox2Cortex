import numpy as np
import nibabel as nib
import os
import sys


def pad_image(input_image, output_image, x_size, y_size, z_size):
    '''
    Pads and/or crops a given 3D/4D NIfTI image to a specified size.
    If the image is already of the desired size, a message is printed
    and no action is performed. Both operations are applied symmetrically
    on both sides of each dimension independently: zero-padding when the
    image is smaller than the target, and center cropping when larger.

    Parameters:
    - input_image (str): Path to the input NIfTI image file.
    - output_image (str): Path where the output NIfTI image will be saved.
    - x_size (int): Desired size along the x-dimension.
    - y_size (int): Desired size along the y-dimension.
    - z_size (int): Desired size along the z-dimension.

    Returns:
    None. The resulting image is saved to the specified output path.
    '''
    nii = nib.load(input_image)
    # Reorient to RAS
    nii = nib.as_closest_canonical(nii)

    # Load data (use dataobj to avoid forcing float64); works for 3D/4D
    data = np.asanyarray(nii.dataobj)
    if data.ndim not in (3, 4):
        raise ValueError(f"Only 3D/4D images are supported; got ndim={data.ndim}")

    in_shape = data.shape[:3]
    target = (x_size, y_size, z_size)
    if in_shape == target:
        # print(f"The input image is already of size {x_size}x{y_size}x{z_size}.")
        nib.save(out, output_image)
        return

    # ------------------------------------------------------------------
    # For each spatial dimension decide whether to pad or crop
    # ------------------------------------------------------------------
    crop_before = [0, 0, 0]
    crop_after  = [0, 0, 0]
    pad_before  = [0, 0, 0]
    pad_after   = [0, 0, 0]

    for i, (in_s, tgt_s) in enumerate(zip(in_shape, target)):
        delta = tgt_s - in_s
        if delta > 0:                       # image too small → pad
            pad_before[i] = delta // 2
            pad_after[i]  = delta - delta // 2
        elif delta < 0:                     # image too large → crop
            crop_amount    = -delta
            crop_before[i] = crop_amount // 2
            crop_after[i]  = crop_amount - crop_amount // 2

    # ------------------------------------------------------------------
    # Step 1: Symmetric centre crop
    # ------------------------------------------------------------------
    slices = tuple(
        slice(crop_before[i], in_shape[i] - crop_after[i]) for i in range(3)
    )
    if data.ndim == 4:
        slices += (slice(None),)            # keep all volumes
    data = data[slices]

    # ------------------------------------------------------------------
    # Step 2: Symmetric zero-pad
    # ------------------------------------------------------------------
    pad_spatial = tuple((pad_before[i], pad_after[i]) for i in range(3))
    pad_width   = pad_spatial + (((0, 0),) if data.ndim == 4 else ())
    data = np.pad(data, pad_width, mode='constant', constant_values=0)

    # ------------------------------------------------------------------
    # Update the affine
    # ------------------------------------------------------------------
    # Mapping from new voxel coords to original voxel coords:
    #   old_voxel = new_voxel + (crop_before - pad_before)
    # so the "new→old" homogeneous transform T has:
    shift = np.array(
        [crop_before[i] - pad_before[i] for i in range(3)], dtype=float
    )
    T = np.eye(4, dtype=float)
    T[:3, 3] = shift                        # translation only

    # Copy header to preserve all fields
    hdr    = nii.header.copy()
    s_code = int(hdr['sform_code'])
    q_code = int(hdr['qform_code'])
    A_s    = nii.get_sform()
    A_q    = nii.get_qform()

    out = nib.Nifti1Image(data, nii.affine, header=hdr)

    if A_s is not None and s_code > 0:
        out.set_sform(A_s @ T, s_code)
    if A_q is not None and q_code > 0:
        out.set_qform(A_q @ T, q_code)

    # Save
    nib.save(out, output_image)
    # return out

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python padding.py input_image.nii output_image.nii x_size y_size z_size")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    x_size = int(sys.argv[3])
    y_size = int(sys.argv[4])
    z_size = int(sys.argv[5])

    pad_image(input_image, output_image, x_size, y_size, z_size)