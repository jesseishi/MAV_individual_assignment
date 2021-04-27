# Imports.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the image in greyscale.
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_327.png'), 0)
# img = cv2.imread('img_315_flat.png', 0)


# Make it an integral image.
def make_integral_image(im):
    # Simply take the cumulative sum along both axis.
    # So this only works on 2D greyscale images.
    cumsum = np.cumsum(np.cumsum(im, axis=0), axis=1)

    # Normalize before returning.
    return cumsum / np.linalg.norm(cumsum)


int_img = make_integral_image(img)

# # There isn't really anything you can see here.
# plt.imshow(int_img, 'gray')
# plt.show()

# Define our output vector.
scores = np.zeros_like(img, dtype=float)
height, width = np.shape(img)

# Define what the corners look like. Each index represents a block of pixels here.
tl = np.array([[ 1, -1,  1, -1,  1, -1],
               [-1,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0,  0,  0],
               [-1,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0,  0,  0]])
tr = np.array([[-1,  1, -1,  1, -1,  1],
               [ 0,  0,  0,  0,  0, -1],
               [ 0,  0,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0, -1],
               [ 0,  0,  0,  0,  0,  1]])
ll = np.array([[ 1,  0,  0,  0,  0,  0],
               [-1,  0,  0,  0,  0,  0],
               [ 1,  0,  0,  0,  0,  0],
               [-1,  0,  0,  0,  0,  0],
               [ 1, -1,  1, -1,  1, -1]])
lr = np.array([[ 0,  0,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0, -1],
               [ 0,  0,  0,  0,  0,  1],
               [ 0,  0,  0,  0,  0, -1],
               [-1,  1, -1,  1, -1,  1]])
features_names = ['top left', 'top right', 'lower left', 'lower right']
features = np.array([tl, tr, ll, lr])
final_scores = np.zeros((np.shape(features)[0], height, width))

feature_height, feature_width = np.shape(tl)

for block_size_px in [4, 5, 6]:
    print("Running for block size {}".format(block_size_px))

    # We keep track of the scores that a feature gets with a certain block size, if it scores better for a different
    # block size, we don't add it but replace it. This was found to work better.
    block_size_scores = np.zeros_like(final_scores)

    for row in range(0, height - feature_height * block_size_px):
        for col in range(0, width - feature_width * block_size_px):

            features_sum = np.zeros(np.shape(features)[0])

            # Get the index of the features.
            for (i, j), _ in np.ndenumerate(features[0]):

                # Get each feature's weight of this index.
                feature_weights = features[:, i, j]

                # If all features don't count this block, skip it.
                if (feature_weights == 0).all():
                    continue

                # Calculate the block value using the integral image.
                block_value = (int_img[row + (i+1) * block_size_px, col + (j+1) * block_size_px] +
                               int_img[row + i * block_size_px,     col + j * block_size_px] -
                               int_img[row + (i+1) * block_size_px, col + j * block_size_px] -
                               int_img[row + i * block_size_px,     col + (j+1) * block_size_px])

                # For each feature, add the sum of this block according to their weight.
                features_sum += feature_weights * block_value
                # tl_sum += weight * block_value

            block_size_scores[:, row, col] += features_sum

    # If the score for this block size is better, make it the final score.
    for i, block_size_score in enumerate(block_size_scores):
        if np.max(block_size_score) > np.max(final_scores[i]):
            print("For the {} corner, pixel size {} was better.".format(features_names[i], block_size_px))
            final_scores[i] = block_size_score

fig, axs = plt.subplots(1, 5)
axs[0].imshow(img, 'gray')

for i, (scores, label) in enumerate(zip(final_scores, features_names)):
    axs[1+i].imshow(scores, 'gray')

    # First y, then x because it returns it as (row, col).
    y, x = np.unravel_index(np.argmax(scores, axis=None), scores.shape)
    axs[0].plot(x, y, 'X', label=label)
    axs[1+i].plot(x, y, 'X')
    axs[i+1].set_title(label)

axs[0].legend()
plt.show()
