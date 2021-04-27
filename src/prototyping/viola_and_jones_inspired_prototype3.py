# Imports.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the image in greyscale.
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_327.png'), 0)
# img = cv2.imread('img_315_flat.png', 0)

int_row = np.cumsum(img, axis=0)
int_col = np.cumsum(img, axis=1)

row_match_score_final = np.zeros_like(img, dtype=int)
col_match_score_final = np.zeros_like(img, dtype=int)

line_length_px = 8
n_blocks = 7

for i in range(n_blocks):

    # We alternate between adding and subtracting blocks (like a checkerboard pattern).
    sign = 1 if (i % 2) == 0 else -1

    end_rows   = int_row[(i+1) * line_length_px : (-n_blocks+i)   * line_length_px, :]
    start_rows = int_row[i     * line_length_px : (-n_blocks+i-1) * line_length_px, :]
    # print(np.shape(end_rows))
    # print(np.shape(start_rows))
    # print(np.shape(row_match_score_final[:-(n_blocks+1)*line_length_px, :]))
    row_match_score_final[:-(n_blocks+1)*line_length_px, :] += sign * (end_rows - start_rows)

    end_cols   = int_col[:, (i+1) * line_length_px : (-n_blocks+i)   * line_length_px]
    start_cols = int_col[:, i     * line_length_px : (-n_blocks+i-1) * line_length_px]
    col_match_score_final[:, :-(n_blocks+1)*line_length_px] += sign * (end_cols - start_cols)


# height, width = np.shape(img)
#
# for line_length_px in [6]:
#     for n_blocks in [7]:
#
#         for row in range(0, height - n_blocks * line_length_px):
#             for col in range(0, width - n_blocks * line_length_px):
#
#                 row_match_score = np.sum([(i % 2) * int_row[row + i * line_length_px, col] for i in range(n_blocks)])
#                 col_match_score = np.sum([(i % 2) * int_col[row, col + i * line_length_px] for i in range(n_blocks)])
#
#                 row_match_score_final[row, col] = row_match_score
#                 col_match_score_final[row, col] = col_match_score

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img, 'gray')
axs[1].imshow(row_match_score_final, 'gray')
axs[2].imshow(col_match_score_final, 'gray')

axs[1].set_title('horizontal block pattern')
axs[2].set_title('vertical block pattern')
plt.show()

#
# # Make it an integral image.
# def make_integral_image(im):
#     # Simply take the cumulative sum along both axis.
#     # So this only works on 2D greyscale images.
#     cumsum = np.cumsum(np.cumsum(im, axis=0), axis=1)
#
#     # Normalize before returning.
#     return cumsum / np.linalg.norm(cumsum)
#
#
# int_img = make_integral_image(img)
#
# # # There isn't really anything you can see here.
# # plt.imshow(int_img, 'gray')
# # plt.show()
#
# # Define our output vector.
# scores = np.zeros_like(img, dtype=float)
# height, width = np.shape(img)
#
# # Define what the corners look like. Each index represents a block of pixels here.
# tl = np.array([[ 1, -1,  1, -1,  1, -1],
#                [-1,  0,  0,  0,  0,  0],
#                [ 1,  0,  0,  0,  0,  0],
#                [-1,  0,  0,  0,  0,  0],
#                [ 1,  0,  0,  0,  0,  0]])
# tr = np.array([[-1,  1, -1,  1, -1,  1],
#                [ 0,  0,  0,  0,  0, -1],
#                [ 0,  0,  0,  0,  0,  1],
#                [ 0,  0,  0,  0,  0, -1],
#                [ 0,  0,  0,  0,  0,  1]])
# ll = np.array([[ 1,  0,  0,  0,  0,  0],
#                [-1,  0,  0,  0,  0,  0],
#                [ 1,  0,  0,  0,  0,  0],
#                [-1,  0,  0,  0,  0,  0],
#                [ 1, -1,  1, -1,  1, -1]])
# lr = np.array([[ 0,  0,  0,  0,  0,  1],
#                [ 0,  0,  0,  0,  0, -1],
#                [ 0,  0,  0,  0,  0,  1],
#                [ 0,  0,  0,  0,  0, -1],
#                [-1,  1, -1,  1, -1,  1]])
# features_names = ['top left', 'top right', 'lower left', 'lower right']
# features = np.array([tl, tr, ll, lr])
# final_scores = np.zeros((np.shape(features)[0], height, width))
#
# feature_height, feature_width = np.shape(tl)
#
# for block_size_px in [4, 5, 6]:
#     print("Running for block size {}".format(block_size_px))
#
#     # We keep track of the scores that a feature gets with a certain block size, if it scores better for a different
#     # block size, we don't add it but replace it. This was found to work better.
#     block_size_scores = np.zeros_like(final_scores)
#
#     for row in range(0, height - feature_height * block_size_px):
#         for col in range(0, width - feature_width * block_size_px):
#
#             features_sum = np.zeros(np.shape(features)[0])
#
#             # Get the index of the features.
#             for (i, j), _ in np.ndenumerate(features[0]):
#
#                 # Get each feature's weight of this index.
#                 feature_weights = features[:, i, j]
#
#                 # If all features don't count this block, skip it.
#                 if (feature_weights == 0).all():
#                     continue
#
#                 # Calculate the block value using the integral image.
#                 block_value = (int_img[row + (i+1) * block_size_px, col + (j+1) * block_size_px] +
#                                int_img[row + i * block_size_px,     col + j * block_size_px] -
#                                int_img[row + (i+1) * block_size_px, col + j * block_size_px] -
#                                int_img[row + i * block_size_px,     col + (j+1) * block_size_px])
#
#                 # For each feature, add the sum of this block according to their weight.
#                 features_sum += feature_weights * block_value
#                 # tl_sum += weight * block_value
#
#             block_size_scores[:, row, col] += features_sum
#
#     # If the score for this block size is better, make it the final score.
#     for i, block_size_score in enumerate(block_size_scores):
#         if np.max(block_size_score) > np.max(final_scores[i]):
#             print("For the {} corner, pixel size {} was better.".format(features_names[i], block_size_px))
#             final_scores[i] = block_size_score
#
# fig, axs = plt.subplots(1, 5)
# axs[0].imshow(img, 'gray')
#
# for i, (scores, label) in enumerate(zip(final_scores, features_names)):
#     axs[1+i].imshow(scores, 'gray')
#
#     # First y, then x because it returns it as (row, col).
#     y, x = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
#     axs[0].plot(x, y, 'X', label=label)
#     axs[1+i].plot(x, y, 'X')
#     axs[i+1].set_title(label)
#
# axs[0].legend()
# plt.show()
