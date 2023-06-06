'''
Install opencv:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
'''

import cv2
import numpy as np
from matplotlib import pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--UseRANSAC", type=int, default=0 )
parser.add_argument("--image1", type=str,  default='data/myleft.jpg' )
parser.add_argument("--image2", type=str,  default='data/myright.jpg' )
# parser.add_argument("--image1", type=str,  default='data/921919841_a30df938f2_o.jpg' )
# parser.add_argument("--image2", type=str,  default='data/4191453057_c86028ce1f_o.jpg' )
# parser.add_argument("--image1", type=str,  default='data/7433804322_06c5620f13_o.jpg' )
# parser.add_argument("--image2", type=str,  default='data/9193029855_2c85a50e91_o.jpg' )
args = parser.parse_args()

print(args)


def FM_by_normalized_8_point(pts1,  pts2):
    #F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )
    # comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 
	
    #construct 3D points
    pts1 = np.concatenate((pts1, np.ones([len(pts1), 1])), axis=1)
    pts2 = np.concatenate((pts2, np.ones([len(pts2), 1])), axis=1)

    #centroid of all points
    cen1 = np.sum(pts1, axis=0)/len(pts1)
    cen2 = np.sum(pts2, axis=0)/len(pts2)

    #2D centeroid
    cen1 = cen1[:2]
    cen2 = cen2[:2]

    #mean distance of all the points from centroid
    sum1 = 0
    sum2 = 0
    for i in range(len(pts1)):
        sum1 += np.sqrt((pts1[i,0] - cen1[0])**2 + (pts1[i,1] - cen1[1])**2)
        sum2 += np.sqrt((pts2[i,0] - cen2[0])**2 + (pts2[i,1] - cen2[1])**2)
    meanDis1 = sum1/len(pts1)
    meanDis2 = sum2/len(pts2)

    #normalize the input points
    normMatrix1 = np.array([[np.sqrt(2)/meanDis1, 0, -cen1[0]*(np.sqrt(2)/meanDis1)], [0, np.sqrt(2)/meanDis1, -cen1[1]*(np.sqrt(2)/meanDis1)], [0, 0, 1]])
    normMatrix2 = np.array([[np.sqrt(2)/meanDis2, 0, -cen2[0]*(np.sqrt(2)/meanDis2)], [0, np.sqrt(2)/meanDis2, -cen2[1]*(np.sqrt(2)/meanDis2)], [0, 0, 1]])
    pts1 = normMatrix1.dot(pts1.T).T
    pts2 = normMatrix2.dot(pts2.T).T

    #construct the coefficient matrix of the linear system
    A = np.ones([len(pts1), 9])
    for i in range(0, len(pts1)):
        A[i][0] = pts1[i][0]*pts2[i][0]
        A[i][1] = pts1[i][0]*pts2[i][1]
        A[i][2] = pts1[i][0]
        A[i][3] = pts1[i][1]*pts2[i][0]
        A[i][4] = pts1[i][1]*pts2[i][1]
        A[i][5] = pts1[i][1]
        A[i][6] = pts2[i][0]
        A[i][7] = pts2[i][1]
        A[i][8] = 1

    #eigen decomposition for (A.T)A
    Eigenvalue, v = np.linalg.eigh(A.T.dot(A))       #obtain eigenvalue and V
    indices = np.argsort(Eigenvalue)[::-1]           #sort eigenvalues
    Eigenvalue = np.sort(Eigenvalue)[::-1]
    value_mask = np.zeros([9, A.shape[0]])
    Eigenvalue = Eigenvalue[:9]
    v = v[:,indices]
    vh = v.T
    Eigenvalue = np.sqrt(Eigenvalue)                 #root of eigenvalue
    sigma_inv = np.linalg.inv(np.diag(Eigenvalue))
    for i in range(min(A.shape[0], A.shape[1])):
        value_mask[i,i] = sigma_inv[i,i]
    u = A.dot(v.T).dot(value_mask)                   #U=A(V.T)sigma_inv

    F_matrix = vh.T[:, 8].reshape(3,3).T             #the least eigenvector

    
    #second decomposition: eigen decompostion for F(F.T)
    Eigenvalue2, u2 = np.linalg.eigh(F_matrix.dot(F_matrix.T))    #obtain eigenvalue and U
    indices2 = np.argsort(Eigenvalue2)[::-1]         #sort eigenvalues
    Eigenvalue2 = np.sort(Eigenvalue2)[::-1]
    value_mask2 = np.zeros([3, 3])
    Eigenvalue2 = Eigenvalue2[:3]
    u2 = u2[:,indices2]
    Eigenvalue2 = np.sqrt(Eigenvalue2)               ##root of eigenvalue
    sigma_inv2 = np.linalg.inv(np.diag(Eigenvalue2))
    for i in range(3):
        value_mask2[i,i] = sigma_inv2[i,i]
    vh2 = value_mask2.dot(u2.T).dot(F_matrix)        #V.T=sigma_inv(U.T)F
    s2 = Eigenvalue2
    
    #enforce the internal constrain
    s2 = np.diag([s2[0], s2[1], 0])
    F_matrix = u2.dot(s2).dot(vh2)

    #de-normalize
    F_matrix = normMatrix2.T.dot(F_matrix).dot(normMatrix1)
    F_matrix = F_matrix/F_matrix[2,2] 

    return F_matrix
    # F:  fundmental matrix
    #return  F


def FM_by_RANSAC(pts1,  pts2):
    #F, mask = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_RANSAC )	
    # comment out the above line of code. 
	
    # Your task is to implement the algorithm by yourself.
    # Do NOT copy&paste any online implementation. 

    #number of keypoints
    length = len(pts1)
    mask = np.zeros(length)
    F = None

    #construct 3D points
    pts1_3D = np.concatenate((pts1, np.ones([len(pts1), 1])), axis=1)
    pts2_3D = np.concatenate((pts2, np.ones([len(pts1), 1])), axis=1)

    max_inlier_num = 0

    #calculate Fundamental Matrix and mask
    for i in range(1500):             #iteration numbers
        ptsIndex = np.random.choice(range(length), size=8, replace=False)         #sample randomly 8 points
        tem_pts1 = []
        tem_pts2 = []
        for j in range(8):
            tem_pts1.append(pts1[ptsIndex[j]])
            tem_pts2.append(pts2[ptsIndex[j]])
        
        #8 pairs of points
        tem_pts1 = np.float32(tem_pts1)
        tem_pts2 = np.float32(tem_pts2)

        F_temp = FM_by_normalized_8_point(tem_pts1, tem_pts2)

        temp_mask = np.zeros(length)
        thresh = 1.3          #threshold
        count = 0             #count inlier numbers
        for k in range(length):
            #calculate epipolar line and its vector
            epipolar_line = F_temp.dot(pts1_3D[k].T)
            epipolar_vec = np.array([-epipolar_line[1], epipolar_line[0]])
            #calculate the distance between points on the 2nd image and the epipolar line
            distance = np.abs(epipolar_line.dot(pts2_3D[k].T))/np.linalg.norm(epipolar_vec)

            #count inlier numbers and calculate mask
            if distance < thresh:
                temp_mask[k] = 1
                count += 1

        #update the max_inlier_num, F and mask
        if count > max_inlier_num:
            max_inlier_num = count
            F = F_temp
            mask = temp_mask
    # F:  fundmental matrix
    # mask:   whetheter the points are inliers
    return  F, mask

	
img1 = cv2.imread(args.image1,0) 
img2 = cv2.imread(args.image2,0)
#print(args.image1)
#print(args.image2)

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
		
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

F = None
if args.UseRANSAC:
    #F,  mask = FM_by_RANSAC(pts1,  pts2)
    F,  mask = cv2.findFundamentalMat(pts1,  pts2, cv2.FM_RANSAC)  #built-in RANSAC function
    # print("error: ")
    # print(np.sum(abs(F-F1)))
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]	
else:
    F = FM_by_normalized_8_point(pts1,  pts2)
    #F, _ = cv2.findFundamentalMat(pts1,pts2,  cv2.FM_8POINT )     #built-in 8POINT function
# print(F)


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2
	
	
# Find epilines corresponding to points in second image,  and draw the lines on first image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,  F)
lines1 = lines1.reshape(-1,3)
img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img6)
plt.show()

# Find epilines corresponding to points in first image, and draw the lines on second image
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
lines2 = lines2.reshape(-1,3)
img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
plt.subplot(121),plt.imshow(img4)
plt.subplot(122),plt.imshow(img3)
plt.show()
