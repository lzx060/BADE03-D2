from utils import dehomogenize, homogenize, draw_epipolar, visualize_pcd
import numpy as np
import cv2
import pdb
import os


def find_fundamental_matrix(shape, pts1, pts2):
    """
    Computes the Fundamental Matrix F that relates points in two images.

    Inputs:
    - shape: Tuple containing shape of img1
    - pts1: Numpy array of shape (N,2) giving image coordinates in img1
    - pts2: Numpy array of shape (N,2) giving image coordinates in img2

    Returns:
    - F: Numpy array of shape (3,3) giving the fundamental matrix F
    """
    # Use OpenCV to calculate the fundamental matrix using the 8-point algorithm
    FOpenCV, _ = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

    # Normalize F by dividing by the last entry to match the OpenCV results
    F = FOpenCV / FOpenCV[2, 2]

    return F


def compute_epipoles(F):
    """
    Given a Fundamental Matrix F, return the epipoles in homogeneous coordinates.
    Check: e2@F and F@e1 should be close to [0,0,0]

    Inputs:
    - F: The fundamental matrix

    Returns:
    - e1: The epipole for image 1 in homogeneous coordinates
    - e2: The epipole for image 2 in homogeneous coordinates
    """
    # Compute the right null space of F (epipole in image 1)
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]  # The last row of Vt is the epipole for image 1

    # Compute the right null space of F^T (epipole in image 2)
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]  # The last row of Vt is the epipole for image 2

    return e1, e2



def find_triangulation(K1, K2, F, pts1, pts2):
    """
    Extracts 3D points from 2D points and camera matrices.

    Inputs:
    - K1: Camera intrinsic matrix for image 1 (3x3)
    - K2: Camera intrinsic matrix for image 2 (3x3)
    - F: Fundamental matrix (3x3)
    - pts1: 2D points in image 1 (Nx2)
    - pts2: 2D points in image 2 (Nx2)

    Returns:
    - pcd: 3D points (Nx4), homogeneous coordinates
    """
    # Number of points
    N = len(pts1)
    pcd = []

    # Calculate camera projection matrices
    M1 = np.dot(K1, np.hstack([np.eye(3), np.zeros((3, 1))]))  # Camera 1 matrix
    M2 = np.dot(K2, np.hstack([np.eye(3), np.zeros((3, 1))]))  # Camera 2 matrix

    # Perform triangulation
    for i in range(N):
        p1 = np.hstack([pts1[i], 1])  # Convert to homogeneous coordinates
        p2 = np.hstack([pts2[i], 1])  # Convert to homogeneous coordinates

        # Construct the matrix A for DLT
        A = np.zeros((4, 4))
        A[0] = p1[0] * M1[2] - M1[0]
        A[1] = p1[1] * M1[2] - M1[1]
        A[2] = p2[0] * M2[2] - M2[0]
        A[3] = p2[1] * M2[2] - M2[1]

        # Solve for X using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pcd.append(X / X[3])  # Normalize to get homogeneous coordinates

    pcd = np.array(pcd)
    return pcd



if __name__ == '__main__':

    # You can run it on one or all the examples
    names = os.listdir("task23")
    output = "results/"

    if not os.path.exists(output):
        os.mkdir(output)

    for name in names:
        print(name)

        # load the information
        img1 = cv2.imread(os.path.join("task23", name, "im1.png"))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(os.path.join("task23", name, "im2.png"))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        data = np.load(os.path.join("task23", name, "data.npz"))
        pts1 = data['pts1'].astype(float)
        pts2 = data['pts2'].astype(float)
        K1 = data['K1']
        K2 = data['K2']
        shape = img1.shape

        # compute F
        F = find_fundamental_matrix(shape, pts1, pts2)
        # compute the epipoles
        e1, e2 = compute_epipoles(F)
        print(e1, e2)
        #to get the real coordinates, divide by the last entry
        print(e1[:2]/e1[-1], e2[:2]/e2[-1])

        outname = os.path.join(output, name + "_us.png")
        # If filename isn't provided or is None, this plt.shows().
        # If it's provided, it saves it
        draw_epipolar(img1, img2, F, pts1[::10, :], pts2[::10, :],
                      epi1=e1, epi2=e2, filename=outname)

        if 0:
            #you can turn this on or off
            pcd = find_triangulation(K1, K2, F, pts1, pts2)
            visualize_pcd(pcd,
                          filename=os.path.join(output, name + "_rec.png"))


