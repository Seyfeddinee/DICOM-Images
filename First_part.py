import os
import matplotlib.pyplot as plt

import matplotlib.colors
import numpy as np
import pydicom
import scipy.ndimage

from matplotlib import animation

# Path to the segmentation file
segmentation_path = "/Users/seyfeddinemeski/Desktop/prj_Medical_image/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/300.000000-Segmentation-91221/1-1.dcm"
segmentation_dataset = pydicom.dcmread(segmentation_path)
segmentation_array = segmentation_dataset.pixel_array



# Path to the data files
data_path = "/Users/seyfeddinemeski/Desktop/prj_Medical_image/HCC_009/02-15-1998-NA-PP CAPLIVER PROTO.-10975/4.000000-Recon 2 LIVER 3 PHASE CAP-85275"
pixel_data = []
files_names = sorted(os.listdir(data_path))
for filename in files_names:
    path = os.path.join(data_path, filename)
    dataset = pydicom.dcmread(path)
    pixel_data.append(dataset.pixel_array)

pixel_len_mm = [4, 0.70, 0.70]
dcm = np.array(pixel_data)

# Normalise DCM
max_val = np.max(dcm)
normalized = dcm / max_val
normalized = np.round(normalized, 2) * 100
normalized_arr = normalized.astype(int)

segmentation_array = np.flip(segmentation_array, axis=1)

img_slice = np.array(normalized_arr[:, :, :])
mask_slice = np.array(segmentation_array[85:145][:, :, :])
seg_img_dcm = []

# Generate segmented images
for i in range(mask_slice.shape[0]):
    cmap = plt.get_cmap('bone')
    norm = plt.Normalize(vmin=np.amin(img_slice[i]), vmax=np.amax(img_slice[i]))
    img_normalized = cmap(norm(img_slice[i]))[..., :3]
    mask_colored = np.stack([mask_slice[i], np.zeros_like(mask_slice[i]), np.zeros_like(mask_slice[i])], axis=-1)
    fusion = 0.8 * img_normalized + 0.2 * mask_colored
    seg_img_dcm.append(fusion[..., 0])

seg_img_dcm = np.array(seg_img_dcm)

# Median planes View
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Sagittal plane
ax[0].imshow(seg_img_dcm[:, :, seg_img_dcm.shape[1] // 2], cmap="bone", alpha=1,
                        aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[0].set_title("Sagittal plane")
# Coronal plane
ax[1].imshow(seg_img_dcm[:, seg_img_dcm.shape[2] // 2, :], cmap="bone", alpha=1,
                        aspect=pixel_len_mm[0] / pixel_len_mm[2])
ax[1].set_title("Coronal plane")
fig.suptitle("Median planes")
plt.show()

# Sagittal planes View

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
# Median plane
ax[0].imshow(seg_img_dcm[:, :, seg_img_dcm.shape[1] // 2], cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[0].set_title("Median")
# Maximum Intensity Projection plane
ax[1].imshow(np.max(seg_img_dcm, axis=2), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[1].set_title("Maximum Intensity Projection")
# Average Intensity Projection plane
ax[2].imshow(np.mean(seg_img_dcm, axis=2), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[2].set_title("Average Intensity Projection")
fig.suptitle("Sagittal planes")
plt.show()

# Sagittal planes
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

# Median plane
ax[0].imshow(seg_img_dcm[:, seg_img_dcm.shape[2] // 2, :], cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[0].set_title("Median")

# Maximum Intensity Projection plane
ax[1].imshow(np.max(seg_img_dcm, axis=1), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[1].set_title("Maximum Intensity Projection")

# Average Intensity Projection plane
ax[2].imshow(np.mean(seg_img_dcm, axis=1), cmap="bone", aspect=pixel_len_mm[0] / pixel_len_mm[1])
ax[2].set_title("Average Intensity Projection")

fig.suptitle("Coronal")
plt.show()

# Generate sagittal plane projections with varying rotation angles
img_min_seg = np.amin(seg_img_dcm)
img_max_seg = np.amax(seg_img_dcm)
cm = matplotlib.colormaps["bone"]

fig, ax = plt.subplots()
os.makedirs("venv/results/", exist_ok=True)

Number_projections = 69

projections_seg = []

# Generate projections
for idx, seg in enumerate(np.linspace(0, 360 * (Number_projections - 1) / Number_projections, num=Number_projections)):
    rotated_img_seg = scipy.ndimage.rotate(seg_img_dcm, seg, axes=(1, 2), reshape=False)
    proj_seg = np.max(rotated_img_seg, axis=2)

    ax.imshow(proj_seg, cmap=cm, vmin=img_min_seg, vmax=img_max_seg, aspect=pixel_len_mm[0] / pixel_len_mm[1], )

    plt.savefig(f"results/Projection_{idx}.png")
    projections_seg.append(proj_seg)

# Create the GIF image
animation_data = [[ax.imshow(img_seg, animated=True, cmap=cm, vmin=img_min_seg, vmax=img_max_seg, aspect=pixel_len_mm[0] / pixel_len_mm[1], )] for img_seg in projections_seg]
anime = animation.ArtistAnimation(fig, animation_data, interval=30, blit=True)
anime.save("results/Animation_coronal_sagittal.gif")
plt.show()
