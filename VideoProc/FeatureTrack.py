import numpy as np 
import cv2

image=cv2.imread('test2.png')
image = cv2.resize(image, (image.shape[1], image.shape[0]))
image_gray = cv2.cvtColor(image, cv2.cv.CV_RGB2GRAY)

features = cv2.goodFeaturesToTrack(image_gray, 50, .01, 50)
features = features.reshape((-1, 2))
print features
for x, y in features:
    cv2.circle(image, (x, y), 10, (0, 0, 255))
cv2.imshow("window name", image)
cv2.waitKey(100000)




# import numpy as np
# import cv2
# from matplotlib import pyplot as plt

# img1 = cv2.imread('IMG_6 (2).jpg',0)          # queryImage
# img2 = cv2.imread('IMG_6 (3).jpg',0) # trainImage
# img3 = cv2.imread('IMG_6 (4).jpg',0)
# img1 = cv2.resize(img1, (img1.shape[1] / 6, img1.shape[0] / 6))
# img2 = cv2.resize(img2, (img2.shape[1] / 6, img2.shape[0] / 6))
# img3 = cv2.resize(img3, (img3.shape[1] / 6, img3.shape[0] / 6))




# # Initiate SIFT detector
# # orb = cv2.ORB()
# orb= cv2.SIFT()

# # find the keypoints and descriptors with SIFT
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)
# kp3, des3 = orb.detectAndCompute(img3,None)



# # create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# bf = cv2.BFMatcher()
# # matches = bf.knnMatch(des1,des2, k=2)

# # Match descriptors.
# # matches12 = bf.match(des1,des2,k=2)
# # matches23 = bf.match(des2,des3,k=2)
# # matches31 = bf.match(des3,des1,k=2)

# matches12 = bfb.knnMatch(des1,des2,k=2)
# matches23 = bf.knnMatch(des2,des3,k=2)
# matches31 = bf.knnMatch(des3,des1,k=2)

# # Sort them in the order of their distance.
# # matches23 = sorted(matches23, key = lambda x:x.distance)
# # matches31 = sorted(matches31, key = lambda x:x.distance)

# print matches12

# good = []
# for m,n in matches12:
#     if m.distance < 0.75*n.distance:
#         good.append(m)

# good = sorted(good, key = lambda x:x.distance)


# def drawMatches(img1, kp1, img2, kp2, matches):
#     """
#     My own implementation of cv2.drawMatches as OpenCV 2.4.9
#     does not have this function available but it's supported in
#     OpenCV 3.0.0

#     This function takes in two images with their associated 
#     keypoints, as well as a list of DMatch data structure (matches) 
#     that contains which keypoints matched in which images.

#     An image will be produced where a montage is shown with
#     the first image followed by the second image beside it.

#     Keypoints are delineated with circles, while lines are connected
#     between matching keypoints.

#     img1,img2 - Grayscale images
#     kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
#               detection algorithms
#     matches - A list of matches of corresponding keypoints through any
#               OpenCV keypoint matching algorithm
#     """

#     # Create a new output image that concatenates the two images together
#     # (a.k.a) a montage
#     rows1 = img1.shape[0]
#     cols1 = img1.shape[1]
#     rows2 = img2.shape[0]
#     cols2 = img2.shape[1]

#     out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

#     # Place the first image to the left
#     out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

#     # Place the next image to the right of it
#     out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

#     # For each pair of points we have between both images
#     # draw circles, then connect a line between them
#     for mat in matches:

#         # Get the matching keypoints for each of the images
#         img1_idx = mat.queryIdx
#         img2_idx = mat.trainIdx

#         # x - columns
#         # y - rows
#         (x1,y1) = kp1[img1_idx].pt
#         (x2,y2) = kp2[img2_idx].pt

#         # Draw a small circle at both co-ordinates
#         # radius 4
#         # colour blue
#         # thickness = 1
#         cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
#         cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

#         # Draw a line in between the two points
#         # thickness = 1
#         # colour blue
#         cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)


#     # Show the image
#     cv2.imshow('Matched Features', out)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




# # print matches
# # cv2.drawm
# # Draw first 10 matches.
# drawMatches(img1,kp1,img2,kp2,good[:10])
# drawMatches(img2,kp2,img3,kp3,matches23[:10])
# drawMatches(img3,kp3,img1,kp1,matches31[:10])

# # plt.imshow(img3),plt.show()