import argparse

import imagesize

import numpy as np

import os

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

parser.add_argument(
    '--base_path', type=str, required=True,
    help='path to MegaDepth'
)
parser.add_argument(
    '--scene_id', type=str, required=True,
    help='scene ID'
)

parser.add_argument(
    '--output_path', type=str, required=True,
    help='path to the output directory'
)

args = parser.parse_args()

base_path = args.base_path
# Remove the trailing / if need be.
if base_path[-1] in ['/', '\\']:
    base_path = base_path[: - 1]
scene_id = args.scene_id

### If want to calc and not load
# base_depth_path = os.path.join(
#     base_path, 'depth_undistorted' #'phoenix/S6/zl548/MegaDepth_v1'
# )
# base_undistorted_sfm_path = os.path.join(
#     base_path, 'Undistorted_SfM'
# )

# undistorted_sparse_path = os.path.join(
#     base_undistorted_sfm_path, scene_id, 'sparse-txt'
# )
# if not os.path.exists(undistorted_sparse_path):    
#     print('No sparse-txt directory for scene %s' % scene_id, undistorted_sparse_path)
#     exit()

# depths_path = os.path.join(
#     base_depth_path, scene_id # , 'dense0', 'depths'
# )
# if not os.path.exists(depths_path):
#     print('No depths directory for scene %s' % scene_id, depths_path)
#     exit()

# images_path = os.path.join(
#     base_undistorted_sfm_path, scene_id, 'images'
# )
# if not os.path.exists(images_path):
#     print('No images directory for scene %s' % scene_id, images_path)
#     exit()

# """
# # Camera list with one line of data per camera:
# #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
# # Number of cameras: 3803
# 8805 PINHOLE 1280 960 2807.16 2808.34 640 480
# 8803 PINHOLE 1280 941 1280.41 1280.17 640 470.5
# """

# # Process cameras.txt
# print('Processing cameras.txt')
# with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
#     raw = f.readlines()[3 :]  # skip the header

# camera_intrinsics = {}
# for camera in raw:
#     camera = camera.split(' ')
#     camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

# """
# head -n 5 points3D.txt 
# # 3D point list with one line of data per point:
# #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
# # Number of points: 225128, mean track length: 19.4148
# 416894 -38.9224 9.16377 18.9185 227 231 231 0.855131 5384 311 5604 5795 8816 5405 2261 6154 4025 2355 4902 10753 8805 7450 10742 9291 4895 2593
# """

# print('Processing points3D.txt')
# # Process points3D.txt
# with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
#     raw = f.readlines()[3 :]  # skip the header

# points3D = {}
# for point3D in raw:
#     point3D = point3D.split(' ')
#     points3D[int(point3D[0])] = np.array([
#         float(point3D[1]), float(point3D[2]), float(point3D[3])
#     ])
    
# """
# # Image list with two lines of data per image:
# #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
# #   POINTS2D[] as (X, Y, POINT3D_ID)
# # Number of images: 3803, mean observations per image: 1149.31
# 8302 0.937158 0.348838 -0.00681464 0.00115301 -0.126252 3.78765 2.77988 8302 3410025276_efcf668cf8_o.jpg
# """

# print('Processing images.txt')
# # Process images.txt
# with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
#     raw = f.readlines()[4 :]  # skip the header

# image_id_to_idx = {}
# image_names = []
# raw_pose = []
# camera = []
# points3D_id_to_2D = []
# n_points3D = []

# print('Processing and all the data')
# for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])):
#     image = image.split(' ')
#     points = points.split(' ')

#     image_id_to_idx[int(image[0])] = idx

#     image_name = image[-1].strip('\n')
#     image_names.append(image_name)

#     raw_pose.append([float(elem) for elem in image[1 : -2]])
#     camera.append(int(image[-2]))
#     current_points3D_id_to_2D = {}
#     for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
#         if int(point3D_id) == -1:
#             continue
#         current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
#     points3D_id_to_2D.append(current_points3D_id_to_2D)
#     n_points3D.append(len(current_points3D_id_to_2D))
# n_images = len(image_names)

# # Image and depthmaps paths
# image_paths = []
# depth_paths = []
# for image_name in image_names:
#     image_path = os.path.join(images_path, image_name)
   
#     # Path to the depth file
#     depth_path = os.path.join(
#         depths_path, '%s.h5' % os.path.splitext(image_name)[0]
#     )
    
#     if os.path.exists(depth_path):
#         # Check if depth map or background / foreground mask
#         file_size = os.stat(depth_path).st_size
#         # Rough estimate - 75KB might work as well
#         if file_size < 100 * 1024:
#             depth_paths.append(None)
#             image_paths.append(None)
#         else:
#             depth_paths.append(depth_path[len(base_path) + 1 :])
#             image_paths.append(image_path[len(base_path) + 1 :])
#     else:
#         depth_paths.append(None)
#         image_paths.append(None)

# # Camera configuration
# intrinsics = []
# poses = []
# principal_axis = []
# points3D_id_to_ndepth = []
# for idx, image_name in enumerate(image_names):
#     if image_paths[idx] is None:
#         intrinsics.append(None)
#         poses.append(None)
#         principal_axis.append([0, 0, 0])
#         points3D_id_to_ndepth.append({})
#         continue
#     image_intrinsics = camera_intrinsics[camera[idx]]
#     K = np.zeros([3, 3])
#     K[0, 0] = image_intrinsics[2]
#     K[0, 2] = image_intrinsics[4]
#     K[1, 1] = image_intrinsics[3]
#     K[1, 2] = image_intrinsics[5]
#     K[2, 2] = 1
#     intrinsics.append(K)

#     image_pose = raw_pose[idx]
#     qvec = image_pose[: 4]
#     qvec = qvec / np.linalg.norm(qvec)
#     w, x, y, z = qvec
#     R = np.array([
#         [
#             1 - 2 * y * y - 2 * z * z,
#             2 * x * y - 2 * z * w,
#             2 * x * z + 2 * y * w
#         ],
#         [
#             2 * x * y + 2 * z * w,
#             1 - 2 * x * x - 2 * z * z,
#             2 * y * z - 2 * x * w
#         ],
#         [
#             2 * x * z - 2 * y * w,
#             2 * y * z + 2 * x * w,
#             1 - 2 * x * x - 2 * y * y
#         ]
#     ])
#     principal_axis.append(R[2, :])
#     t = image_pose[4 : 7]
#     # World-to-Camera pose
#     current_pose = np.zeros([4, 4])
#     current_pose[: 3, : 3] = R
#     current_pose[: 3, 3] = t
#     current_pose[3, 3] = 1
#     # Camera-to-World pose
#     # pose = np.zeros([4, 4])
#     # pose[: 3, : 3] = np.transpose(R)
#     # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
#     # pose[3, 3] = 1
#     poses.append(current_pose)
    
#     current_points3D_id_to_ndepth = {}
#     for point3D_id in points3D_id_to_2D[idx].keys():
#         p3d = points3D[point3D_id]
#         current_points3D_id_to_ndepth[point3D_id] = (np.dot(R[2, :], p3d) + t[2]) / (.5 * (K[0, 0] + K[1, 1])) 
#     points3D_id_to_ndepth.append(current_points3D_id_to_ndepth)
# principal_axis = np.array(principal_axis)
# angles = np.rad2deg(np.arccos(
#     np.clip(
#         np.dot(principal_axis, np.transpose(principal_axis)),
#         -1, 1
#     )
# ))

# # Compute overlap score
# overlap_matrix = np.full([n_images, n_images], -1.)
# scale_ratio_matrix = np.full([n_images, n_images], -1.)
# for idx1 in range(n_images):
#     if image_paths[idx1] is None or depth_paths[idx1] is None:
#         continue
#     for idx2 in range(idx1 + 1, n_images):
#         if image_paths[idx2] is None or depth_paths[idx2] is None:
#             continue
#         matches = (
#             points3D_id_to_2D[idx1].keys() &
#             points3D_id_to_2D[idx2].keys()
#         )
#         min_num_points3D = min(
#             len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
#         )
#         overlap_matrix[idx1, idx2] = len(matches) / len(points3D_id_to_2D[idx1])  # min_num_points3D
#         overlap_matrix[idx2, idx1] = len(matches) / len(points3D_id_to_2D[idx2])  # min_num_points3D
#         if len(matches) == 0:
#             continue
#         points3D_id_to_ndepth1 = points3D_id_to_ndepth[idx1]
#         points3D_id_to_ndepth2 = points3D_id_to_ndepth[idx2]
#         nd1 = np.array([points3D_id_to_ndepth1[match] for match in matches])
#         nd2 = np.array([points3D_id_to_ndepth2[match] for match in matches])
#         min_scale_ratio = np.min(np.maximum(nd1 / nd2, nd2 / nd1))
#         scale_ratio_matrix[idx1, idx2] = min_scale_ratio
#         scale_ratio_matrix[idx2, idx1] = min_scale_ratio

# overlap_matrix = np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{scene_id}_calcoverlap_matrix.npy", overlap_matrix, allow_pickle=True)


############ generic debugging stuff #################
# print(f"image_paths={image_paths}")
# print(f"depth_paths={depth_paths}")
# print(f"intrinsics={intrinsics}")
# print(f"poses={poses}")
# print(f"overlap_matrix={overlap_matrix}")
# print(f"scale_ratio_matrix={scale_ratio_matrix}")
# print(f"angles={angles}")
# print(f"n_points3D={n_points3D}")
# print(f"points3D_id_to_2D={points3D_id_to_2D}")
# print(f"points3D_id_to_ndepth={points3D_id_to_ndepth}")

# try:
#     print(f"image_paths shape: {len(image_paths)}")
#     print(f"depth_paths shape: {len(depth_paths)}")
#     print(f"intrinsics shape: {len(intrinsics)}")
#     print(f"poses shape: {len(poses)}")
#     print(f"overlap_matrix shape: {overlap_matrix.shape}")
#     print(f"scale_ratio_matrix shape: {scale_ratio_matrix.shape}")
#     print(f"angles shape: {angles.shape}")
#     print(f"n_points3D shape: {len(n_points3D)}")
#     print(f"points3D_id_to_2D shape: {len(points3D_id_to_2D)}")
#     print(f"points3D_id_to_ndepth shape: {len(points3D_id_to_ndepth)}")    
# except:
#     pass

# print(any(x is None for x in intrinsics))
# print(any(x is None for x in poses))

# count = 20
# print(f"image_paths={image_paths[:count]}")
# print(f"depth_paths={depth_paths[:count]}")
# print(f"intrinsics={intrinsics[:count]}")

# print(overlap_matrix.shape)
# print(overlap_matrix)
# for i in range(n_images):
#     for j in range(i + 1, n_images):
#         if overlap_matrix[i, j] != -1:
#             print(f"Processing overlap between images {i} and {j}, {overlap_matrix[i, j]}")




# if os.path.exists(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{scene_id}_calcoverlap_matrix.npy"):
#     overlap_matrix = np.load(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{scene_id}_calcoverlap_matrix.npy", allow_pickle=True)
# else:
# overlap_matrix = np.save(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{scene_id}_calcoverlap_matrix.npy", overlap_matrix, allow_pickle=True)
#################################################


############ plotting and comparison code #################
overlap_matrix = np.load(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{scene_id}_calcoverlap_matrix.npy", allow_pickle=True)

to_save = "0001"
# to_save = scene_id
loaded_overlap = np.load(f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/{to_save}_overlap_matrix.npy", allow_pickle=True)

path = "/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/0001_overlap_matrix.npy"
secondTitle = "truth"

titleStr = path.split("/")[-1].split(".")[0] + " vs " + secondTitle
expNum = "withMegadepthCode"

# Assume 'calculated_matrix' and 'ground_truth_matrix' are your numpy arrays for the matrices
calculated_matrix = overlap_matrix  #np.array(calculated_overlap_matrix)
ground_truth_matrix = loaded_overlap #np.array(ground_truth_overlap_matrix)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
cmap = "viridis"  # Color map for visualization

# Plot calculated overlap matrix
cax = ax[0].imshow(calculated_matrix, cmap=cmap, interpolation='nearest')
ax[0].set_title('Calculated Overlap Matrix')
fig.colorbar(cax, ax=ax[0])

# Plot ground truth overlap matrix
gax = ax[1].imshow(ground_truth_matrix, cmap=cmap, interpolation='nearest')
ax[1].set_title('Ground Truth Overlap Matrix')
fig.colorbar(gax, ax=ax[1])

# Save the figure
plt.title(titleStr)
plt.savefig(f'/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/overlap_matrices{expNum}.png')
plt.close()  # Close the plot to free up memory

print("Calculated Matrix Stats:")
print("Mean:", np.mean(calculated_matrix))
print("Median:", np.median(calculated_matrix))
print("Std Deviation:", np.std(calculated_matrix))

print("Ground Truth Matrix Stats:")
print("Mean:", np.mean(ground_truth_matrix))
print("Median:", np.median(ground_truth_matrix))
print("Std Deviation:", np.std(ground_truth_matrix))

comparison = np.isclose(calculated_matrix, ground_truth_matrix, atol=0.01)

# Find indices where comparison is False and print corresponding values
false_indices = np.where(comparison == False)
if len(false_indices[0]) == 0:
    print("All values are the same.")
else:
    print("Indices and Values where comparison is False:")
    calc_value = calculated_matrix[idx]
    truth_value = ground_truth_matrix[idx]
    print(f"Index: {idx}, Calculated Value: {calc_value}, Ground Truth Value: {truth_value}")

    for idx in zip(false_indices[0], false_indices[1]):
        print(f"Index: {idx}, Value: {calculated_matrix[idx]}")
print(f"Comparison Result (True means close enough): {comparison}")
print()

def count_matching_values(calculated_matrix, ground_truth_matrix):
    """Count the number of matching values in two matrices, excluding -1 and 0."""
    # Ensure the matrices have the same shape
    if calculated_matrix.shape != ground_truth_matrix.shape:
        raise ValueError("Matrices must have the same dimensions.")
    
    count = 0
    # Iterate over all elements in the matrices
    for i in range(calculated_matrix.shape[0]):
        for j in range(calculated_matrix.shape[1]):
            if calculated_matrix[i, j] == ground_truth_matrix[i, j]: #and calculated_matrix[i, j] not in [-1, 0]:
                count += 1
    
    return count

count = count_matching_values(calculated_matrix, ground_truth_matrix)
print("Number of matching values (excluding -1 and 0):", count)

print(f"Plot saved to: /homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/overlap_matrices{expNum}.png")

plt.figure(figsize=(12, 6))
plt.hist(calculated_matrix.flatten(), bins=50, alpha=0.5, label='Calculated')
plt.hist(ground_truth_matrix.flatten(), bins=50, alpha=0.5, label='Ground Truth')
plt.legend()
plt.title("Histogram of Overlap Values")
plt.xlabel("Overlap Score")
plt.ylabel("Frequency")
histpath = f"/homes/tp4618/Documents/bitbucket/SuperGlueThesis/external/d2-net/histogram_{expNum}.png"
plt.savefig(histpath)
plt.close()

print(f"Plot saved to: {histpath}")


#########
# np.savez(
#     os.path.join(args.output_path, '%s.npz' % scene_id),
#     image_paths=image_paths,
#     depth_paths=depth_paths,
#     intrinsics=intrinsics,
#     poses=poses,
#     overlap_matrix=overlap_matrix,
#     scale_ratio_matrix=scale_ratio_matrix,
#     angles=angles,
#     n_points3D=n_points3D,
#     points3D_id_to_2D=points3D_id_to_2D,
#     points3D_id_to_ndepth=points3D_id_to_ndepth
# )


# import argparse

# import imagesize

# import numpy as np

# import os

# parser = argparse.ArgumentParser(description='MegaDepth preprocessing script')

# parser.add_argument(
#     '--base_path', type=str, required=True,
#     help='path to MegaDepth'
# )
# parser.add_argument(
#     '--scene_id', type=str, required=True,
#     help='scene ID'
# )

# parser.add_argument(
#     '--output_path', type=str, required=True,
#     help='path to the output directory'
# )

# args = parser.parse_args()

# base_path = args.base_path
# # Remove the trailing / if need be.
# if base_path[-1] in ['/', '\\']:
#     base_path = base_path[: - 1]
# scene_id = args.scene_id

# base_depth_path = os.path.join(
#     base_path, 'phoenix/S6/zl548/MegaDepth_v1'
# )
# base_undistorted_sfm_path = os.path.join(
#     base_path, 'Undistorted_SfM'
# )

# undistorted_sparse_path = os.path.join(
#     base_undistorted_sfm_path, scene_id, 'sparse-txt'
# )
# if not os.path.exists(undistorted_sparse_path):
#     exit()

# depths_path = os.path.join(
#     base_depth_path, scene_id, 'dense0', 'depths'
# )
# if not os.path.exists(depths_path):
#     exit()

# images_path = os.path.join(
#     base_undistorted_sfm_path, scene_id, 'images'
# )
# if not os.path.exists(images_path):
#     exit()

# # Process cameras.txt
# with open(os.path.join(undistorted_sparse_path, 'cameras.txt'), 'r') as f:
#     raw = f.readlines()[3 :]  # skip the header

# camera_intrinsics = {}
# for camera in raw:
#     camera = camera.split(' ')
#     camera_intrinsics[int(camera[0])] = [float(elem) for elem in camera[2 :]]

# # Process points3D.txt
# with open(os.path.join(undistorted_sparse_path, 'points3D.txt'), 'r') as f:
#     raw = f.readlines()[3 :]  # skip the header

# points3D = {}
# for point3D in raw:
#     point3D = point3D.split(' ')
#     points3D[int(point3D[0])] = np.array([
#         float(point3D[1]), float(point3D[2]), float(point3D[3])
#     ])
    
# # Process images.txt
# with open(os.path.join(undistorted_sparse_path, 'images.txt'), 'r') as f:
#     raw = f.readlines()[4 :]  # skip the header

# image_id_to_idx = {}
# image_names = []
# raw_pose = []
# camera = []
# points3D_id_to_2D = []
# n_points3D = []
# for idx, (image, points) in enumerate(zip(raw[:: 2], raw[1 :: 2])):
#     image = image.split(' ')
#     points = points.split(' ')

#     image_id_to_idx[int(image[0])] = idx

#     image_name = image[-1].strip('\n')
#     image_names.append(image_name)

#     raw_pose.append([float(elem) for elem in image[1 : -2]])
#     camera.append(int(image[-2]))
#     current_points3D_id_to_2D = {}
#     for x, y, point3D_id in zip(points[:: 3], points[1 :: 3], points[2 :: 3]):
#         if int(point3D_id) == -1:
#             continue
#         current_points3D_id_to_2D[int(point3D_id)] = [float(x), float(y)]
#     points3D_id_to_2D.append(current_points3D_id_to_2D)
#     n_points3D.append(len(current_points3D_id_to_2D))
# n_images = len(image_names)

# # Image and depthmaps paths
# image_paths = []
# depth_paths = []
# for image_name in image_names:
#     image_path = os.path.join(images_path, image_name)
   
#     # Path to the depth file
#     depth_path = os.path.join(
#         depths_path, '%s.h5' % os.path.splitext(image_name)[0]
#     )
    
#     if os.path.exists(depth_path):
#         # Check if depth map or background / foreground mask
#         file_size = os.stat(depth_path).st_size
#         # Rough estimate - 75KB might work as well
#         if file_size < 100 * 1024:
#             depth_paths.append(None)
#             image_paths.append(None)
#         else:
#             depth_paths.append(depth_path[len(base_path) + 1 :])
#             image_paths.append(image_path[len(base_path) + 1 :])
#     else:
#         depth_paths.append(None)
#         image_paths.append(None)

# # Camera configuration
# intrinsics = []
# poses = []
# principal_axis = []
# points3D_id_to_ndepth = []
# for idx, image_name in enumerate(image_names):
#     if image_paths[idx] is None:
#         intrinsics.append(None)
#         poses.append(None)
#         principal_axis.append([0, 0, 0])
#         points3D_id_to_ndepth.append({})
#         continue
#     image_intrinsics = camera_intrinsics[camera[idx]]
#     K = np.zeros([3, 3])
#     K[0, 0] = image_intrinsics[2]
#     K[0, 2] = image_intrinsics[4]
#     K[1, 1] = image_intrinsics[3]
#     K[1, 2] = image_intrinsics[5]
#     K[2, 2] = 1
#     intrinsics.append(K)

#     image_pose = raw_pose[idx]
#     qvec = image_pose[: 4]
#     qvec = qvec / np.linalg.norm(qvec)
#     w, x, y, z = qvec
#     R = np.array([
#         [
#             1 - 2 * y * y - 2 * z * z,
#             2 * x * y - 2 * z * w,
#             2 * x * z + 2 * y * w
#         ],
#         [
#             2 * x * y + 2 * z * w,
#             1 - 2 * x * x - 2 * z * z,
#             2 * y * z - 2 * x * w
#         ],
#         [
#             2 * x * z - 2 * y * w,
#             2 * y * z + 2 * x * w,
#             1 - 2 * x * x - 2 * y * y
#         ]
#     ])
#     principal_axis.append(R[2, :])
#     t = image_pose[4 : 7]
#     # World-to-Camera pose
#     current_pose = np.zeros([4, 4])
#     current_pose[: 3, : 3] = R
#     current_pose[: 3, 3] = t
#     current_pose[3, 3] = 1
#     # Camera-to-World pose
#     # pose = np.zeros([4, 4])
#     # pose[: 3, : 3] = np.transpose(R)
#     # pose[: 3, 3] = -np.matmul(np.transpose(R), t)
#     # pose[3, 3] = 1
#     poses.append(current_pose)
    
#     current_points3D_id_to_ndepth = {}
#     for point3D_id in points3D_id_to_2D[idx].keys():
#         p3d = points3D[point3D_id]
#         current_points3D_id_to_ndepth[point3D_id] = (np.dot(R[2, :], p3d) + t[2]) / (.5 * (K[0, 0] + K[1, 1])) 
#     points3D_id_to_ndepth.append(current_points3D_id_to_ndepth)
# principal_axis = np.array(principal_axis)
# angles = np.rad2deg(np.arccos(
#     np.clip(
#         np.dot(principal_axis, np.transpose(principal_axis)),
#         -1, 1
#     )
# ))

# # Compute overlap score
# overlap_matrix = np.full([n_images, n_images], -1.)
# scale_ratio_matrix = np.full([n_images, n_images], -1.)
# for idx1 in range(n_images):
#     if image_paths[idx1] is None or depth_paths[idx1] is None:
#         continue
#     for idx2 in range(idx1 + 1, n_images):
#         if image_paths[idx2] is None or depth_paths[idx2] is None:
#             continue
#         matches = (
#             points3D_id_to_2D[idx1].keys() &
#             points3D_id_to_2D[idx2].keys()
#         )
#         min_num_points3D = min(
#             len(points3D_id_to_2D[idx1]), len(points3D_id_to_2D[idx2])
#         )
#         overlap_matrix[idx1, idx2] = len(matches) / len(points3D_id_to_2D[idx1])  # min_num_points3D
#         overlap_matrix[idx2, idx1] = len(matches) / len(points3D_id_to_2D[idx2])  # min_num_points3D
#         if len(matches) == 0:
#             continue
#         points3D_id_to_ndepth1 = points3D_id_to_ndepth[idx1]
#         points3D_id_to_ndepth2 = points3D_id_to_ndepth[idx2]
#         nd1 = np.array([points3D_id_to_ndepth1[match] for match in matches])
#         nd2 = np.array([points3D_id_to_ndepth2[match] for match in matches])
#         min_scale_ratio = np.min(np.maximum(nd1 / nd2, nd2 / nd1))
#         scale_ratio_matrix[idx1, idx2] = min_scale_ratio
#         scale_ratio_matrix[idx2, idx1] = min_scale_ratio

# np.savez(
#     os.path.join(args.output_path, '%s.npz' % scene_id),
#     image_paths=image_paths,
#     depth_paths=depth_paths,
#     intrinsics=intrinsics,
#     poses=poses,
#     overlap_matrix=overlap_matrix,
#     scale_ratio_matrix=scale_ratio_matrix,
#     angles=angles,
#     n_points3D=n_points3D,
#     points3D_id_to_2D=points3D_id_to_2D,
#     points3D_id_to_ndepth=points3D_id_to_ndepth
# )
