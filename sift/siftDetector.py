# TODO: How the fuck do I find the correct coordinates for the rectangle?


import glob

import cv2
import matplotlib.pyplot as plt

# The list of keypoints in each image (the query image and the testing image)
listKeypoints1 = []
listKeyPoints2 = []

# Reads the template image
templateImage = cv2.imread('template.jpg')

# Converts the colored image from RGB to grayscale
templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

# Selects all image paths in the testing folder
imagePaths = glob.glob('./dataset/testing/*.jpg')

# Loops through all the images in the testing folder
for imagePath in imagePaths:
    # Reads an image
    inputImage = cv2.imread(imagePath)

    # Converts the image from RGB to grayscale
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Creates a sift instance
    sift = cv2.xfeatures2d.SIFT_create()

    # Computes keypoints in the template image
    keypointsTemplateImage, descriptorsTemplateImage = sift.detectAndCompute(templateImage, None)

    # Computes keypoints in the testing image
    keypointsInputImage, descriptorsImputImage = sift.detectAndCompute(inputImage, None)

    # Creates an instance of a BF matcher
    BFMatcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Finds matching keypoints
    matches = BFMatcher.match(descriptorsTemplateImage, descriptorsImputImage)

    # Sorts the matches
    matches = sorted(matches, key=lambda x: x.distance)

    # Loops through all the matching keypoints
    for match in matches:

        # Get the matching keypoints for each of the images
        templateImageIdx = match.queryIdx
        imageIdx = match.trainIdx

        # x is equal to column
        # y is equal to row
        # Get the coordinates
        (x1, y1) = keypointsTemplateImage[templateImageIdx].pt
        (x2, y2) = keypointsInputImage[imageIdx].pt

        # Append to each list
        listKeypoints1.append((x1, y1))
        listKeyPoints2.append((x2, y2))

    # mange kule kordinater
    print(listKeypoints1)
    topLeft = max(listKeypoints1, key=lambda x: (x[0], -x[1]))
    bottomRight = max(listKeypoints1, key=lambda x: (-x[0], x[1]))
    top =  max(listKeypoints1, key=lambda x: (x[0]))
    print(top)
    print (topLeft)
    print (bottomRight)
    x1 = topLeft[0]
    y1 = topLeft[1]
    x2 = bottomRight[0]
    y2 = bottomRight[1]

    resultImage = cv2.drawMatches(templateImage, keypointsTemplateImage, inputImage,
                                  keypointsInputImage, matches[:100], inputImage, flags=2)

    # God help me resultImage = cv2.rectangle(resultImage,(60,8) , (474,109), (0,255,0), thickness=10, lineType=None)

    plt.imshow(resultImage), plt.show()

    cv2.waitKey(0)
