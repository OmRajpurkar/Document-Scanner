import cv2
import numpy as np

#############################
widthImg = 480
heightImg = 640
############################

vid = cv2.VideoCapture(0)
vid.set(3, widthImg)
vid.set(4, heightImg)
vid.set(10, 150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur =  cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDialation = cv2.dilate(imgCanny, kernel, iterations=2)         # This is used to make the edges bolder
    imgTreshold = cv2.erode(imgDialation, kernel, iterations=1)       # This is used to make the edges thinner
    return imgTreshold

def getContours(img):
    biggest = np.array([])
    maxArea = 0

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt) # This is used to find the area of the contour.
        if area>5000: # The areas below 500 pixels will not be considered
            #cv2.drawContours(imgContour, cnt, -1, (255,0,0), 3) # -1 denotes that we need to draw all the contours
            perimeter = cv2.arcLength(cnt, True) # The true indicates that the contour is closed
            approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True) # This method is used to find the approximate number of contours

            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area

    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)  # -1 denotes that we need to draw all the contours
    return biggest

# This is used to reorder the points for the warp perspective
# For the [0,0] it is very simple as we need to find the smallest point
# For [widthImg, heightImg] is also simple as we need to find the largest point
# For [widthImg,0][0,heightImg] we need to subtract them. One will return positive the other will return negative.
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)

    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getWarp(img, biggest):          # This is to change the perspective of the image
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthImg,heightImg))

    return imgOutput


def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver



while True:
    success, img = vid.read()
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    imgTreshold = preProcessing(img)

    biggest = getContours(imgTreshold)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imageArray = ([img, imgTreshold ],
                  [imgContour, imgWarped])
    else:
        imgWarped = img
        imageArray = ([img, imgTreshold],
                      [img, img])

    stackedImages = stackImages(0.5, imageArray)

    cv2.imshow("WorkFlow", stackedImages)
    cv2.imshow("ImageWarped", imgWarped)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
