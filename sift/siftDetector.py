import glob
import csv
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np

# A list of all the iou scores
iouScores = []

# Imports the ground truths from the csv file
with open("GroundTruth-numbered-only.csv", newline='') as csvfile:
    dataList = list(csv.reader(csvfile))

# Sorts the list in this manner: (imageName, x1, y1, x2, y2)
for i in range(len(dataList)):
    dataTemp1 = dataList[i][1]
    dataTemp2 = dataList[i][2]
    dataList[i][1] = dataList[i][3]
    dataList[i][2] = dataTemp1
    dataList[i][3] = dataList[i][4]
    dataList[i][4] = dataTemp2

# Sorts the list after imageName
dataList.sort()

# A function that calculates the intersection over union for two rectangles
def bb_intersection_over_union(boxA, boxB):
    
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the interscation
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# The list of keypoints in each image (the query image and the testing image)
listKeypoints1 = []
listKeyPoints2 = []

# Reads the template image
templateImage = cv2.imread('fish_croped.png')

# Converts the colored image from RGB to grayscale
templateImage = cv2.cvtColor(templateImage, cv2.COLOR_BGR2GRAY)

# Selects all image paths in the testing folder
imagePaths = glob.glob('./dataset/testing/*.jpg')
imagePaths.sort()

# Index for each image in the path
i = 0

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

    # Sharpening the image
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    inputImage = cv2.filter2D(inputImage, -1, filter)

    ###
    # Uncomment this to denoise the image (Takes a lot of time)
    # cv2.fastNlMeansDenoisingColored(inputImage, None, 20,20,14,42)
    ###

    # Converts the image from RGB to grayscale
    inputImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)

    # Creates a sift instance
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=None, nOctaveLayers=None, contrastThreshold=None, edgeThreshold=None, sigma=None)

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

    # Saves the upper left corner and bottom right corner
    topLeft = (int(smallestX), int(smallestY))
    bottomRight = (int(largestX), int(largestY))

    boxA = int(dataList[i][1]), int(dataList[i][2]), int(dataList[i][3]), int(dataList[i][4])
    boxB = (int(smallestX), int(smallestY), int(largestX), int(largestY))

    print("Image path: \t\t" + imagePath)
    print("Ground truth box: \t" + str(boxA))
    print("Sift bounding box: \t" + str(boxB))

    # Gets the iou score from the iou function
    iouScore = bb_intersection_over_union(boxA, boxB)

    # Set the iou score to 0 if it turns out to be negative
    if iouScore < 0:
        iouScore = 0
    
    print("The iou score: \t\t" + str(iouScore))
    print("\n")

    ###
    # Uncomment to draw all the matching keypoints
    # Uncomment this if you want to see all the keypoints
    #outputImage = cv2.drawMatches(templateImage, keypointsTemplateImage, inputImage,
    #keypointsInputImage, matches[:10000], inputImage, flags=2)
    ###

    # Draws the bounding box on top of the image
    outputImage = cv2.imread(imagePath)
    outputImage = cv2.rectangle(
        outputImage,
        topLeft,
        bottomRight,
        (255, 0, 0),
        thickness=2
    )
    
    # Draws the ground truth
    outputImage = cv2.rectangle(
        outputImage,
        (int(dataList[i][1]), int(dataList[i][2])), 
        (int(dataList[i][3]), int(dataList[i][4])), 
        (0,255,0),
        thickness=2
    )

    # Append the iou score the iou scores list
    iouScores.append(iouScore)
    
    # Shows the image (press key/close image to go to the next image)
    plt.imshow(outputImage), plt.show()
    cv2.waitKey(0)

    # Increments by one to get the next image in the dataList
    i = i + 1

print("The iou scores: ")
print(iouScores)
print("\n")
print("Average iou:" + str(sum(iouScores)/50))
print("Median IoU score: " + str(iouScores[25]))
print("The highest iou score: " + str(max(iouScores)))
