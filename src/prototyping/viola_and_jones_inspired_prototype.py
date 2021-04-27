# Imports.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Load the image in greyscale.
data_folder = os.path.abspath(os.path.join(os.curdir, '../../..', 'WashingtonOBRace'))
img = cv2.imread(os.path.join(data_folder, 'img_182.png'), 0)


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

# Loop through all features.
for width_of_blocks in [9]:
    for height_of_blocks in [1]:
        for number_of_blocks in [7]:
            print(width_of_blocks, height_of_blocks, number_of_blocks)

            # Loop through all pixels where we can start.
            for col in np.arange(0,
                                 np.floor(width - width_of_blocks * number_of_blocks),
                                 dtype=int):
                for row in np.arange(0,
                                     np.floor(height - height_of_blocks),
                                     dtype=int):

                    # Calculate the sum for this feature by looping through the blocks.
                    feature_sum = 0
                    for block_i in range(number_of_blocks):

                        # Since we're using an integral image, we can simplify this calculation by a lot.
                        # Do: lr + ul - ur - ll.
                        block_value = (int_img[row + height_of_blocks, col + width_of_blocks * (block_i+1)] +
                                       int_img[row,                    col + width_of_blocks * block_i] -
                                       int_img[row,                    col + width_of_blocks * (block_i+1)] -
                                       int_img[row + height_of_blocks, col + width_of_blocks * block_i])

                        # If this is an odd block, block_i is even and we want to add it.
                        if block_i % 2 == 0:
                            feature_sum += block_value

                        # For an uneven block, subtract.
                        else:
                            feature_sum -= block_value

                    # Now put this feature sum into the score for the pixel in the middle of this feature.
                    scores[row + height_of_blocks // 2, col + (width_of_blocks * number_of_blocks) // 2] += feature_sum


fig, axs = plt.subplots(1, 2)
axs[0].imshow(img, 'gray')
axs[1].imshow(scores, 'gray')
plt.show()
