# TODO: Make not crash when template too big for image

import cv2
import glob

## These imports are useful when you want to plot the figures
import numpy as np
from matplotlib import pyplot as plt


# Saves all the image paths in an array
imagePaths = glob.glob(r"./dataset/testing/*.jpg")
# Load template
template = cv2.imread('template.jpg', 0)

# Saved the width and the height of the template
width, height = template.shape[::-1]

# Save template matching methods
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

# Do this for all images in the testing folder.
for imgPath in imagePaths:
    # Load image
    print(imgPath)
    inputImage = cv2.imread(imgPath, 0)

    # For loop that loops through each method in the methods list
    for meth in methods:
        #### Attempting to make a try catch for when the template is larger than the image (it sucks)
        try:
            # Reads an image from the path list
            inputImage = cv2.imread(imgPath, 0)

            # Takes one of the methods from the method list
            method = eval(meth)

            # Apply template Matching
            result = cv2.matchTemplate(inputImage, template, method)

            minValue, maxValue, minLocation, maxLocation = cv2.minMaxLoc(result)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = minLocation
            else:
                top_left = maxLocation

            # Saves bottom right coordinate
            bottom_right = (top_left[0] + width, top_left[1] + height)

            # Creates rectangle on top of the image
            cv2.rectangle(inputImage, top_left, bottom_right, 255, 2)

            ### Prints the methid used and the coordinates of the rectangle
            print('Method: ' + meth)
            print('Coordinates: '), print(top_left, bottom_right)
            print('\n')

            cv2.imshow(meth, inputImage)

            #### Optional if you want figures and shit
            # fig = plt.figure()
            # plt.subplot(121),plt.imshow(res,cmap = 'gray')
            # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
            # plt.subplot(122),plt.imshow(img,cmap = 'gray')
            # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
            # plt.suptitle(meth)

            # plt.show()
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        except ValueError:
            print('Oooopsie woopsie, no template match found.')
