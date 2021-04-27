# TODO: How the fuck do I find the correct coordinates for the rectangle?


import glob
import sys
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
    # The 'axes'
    largestX = 0
    smallestX = sys.maxsize
    largestY = 0
    smallestY = sys.maxsize

    # The list of keypoints in each image (the query image and the testing image)
    listKeypoints1 = []
    listKeyPoints2 = []

    # Reads an image
    inputImage = cv2.imread(imagePath)

    # Converts the image from RGB to grayscale
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Creates a sift instance
    sift = cv2.xfeatures2d.SIFT_create()

    # Computes keypoints in the template image
    keypointsTemplateImage, descriptorsTemplateImage = sift.detectAndCompute(templateImage, None)

    # Computes keypoints in the testing image
    keypointsInputImage, descriptorsInputImage = sift.detectAndCompute(inputImage, None)

    # Creates an instance of a BF matcher
    BFMatcher = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    # Finds matching keypoints
    matches = BFMatcher.match(descriptorsTemplateImage, descriptorsInputImage)

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

    # Find the largest and smallest axis value for each keypoint match
    for keypoint in listKeyPoints2:
        keypointX = keypoint[0]
        keypointY = keypoint[1]
        if keypointX > largestX:
            largestX = keypointX
        if keypointY > largestY:
            largestY = keypointY
        if keypointX < smallestX:
            smallestX = keypointX
        if keypointY < smallestY:
            smallestY = keypointY
    print(listKeyPoints2)
    print("\n")
    print(largestX)
    print(largestY)
    print(smallestX)
    print(smallestY)

    topLeft = (int(smallestX), int(smallestY))
    bottomRight = (int(largestX), int(largestY))

    # Draws all the matching keypoints
    # Uncomment this if you want to see all the keypoints
    # resultImage = cv2.drawMatches(templateImage, keypointsTemplateImage, inputImage,
    # keypointsInputImage, matches[:100], inputImage, flags=2)

    # Draws the bounding box on top of the image
    outputImage = cv2.imread(imagePath)
    outputImage = cv2.rectangle(
        outputImage,
        topLeft,
        bottomRight,
        (0, 255, 0),
        thickness=2,
        lineType=None
    )

    plt.imshow(outputImage), plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
