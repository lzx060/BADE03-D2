import numpy as np


def find_projection(pts2d, pts3d):
    """
    Computes the camera projection matrix M that goes from world 3D coordinates
    to 2D image coordinates.

    Inputs:
    - pts2d: Numpy array of shape (N, 2), 2D image coordinates
    - pts3d: Numpy array of shape (N, 3), 3D world coordinates

    Returns:
    - M: Numpy array of shape (3, 4), camera projection matrix
    """
    N = pts2d.shape[0]

    # Prepare the matrix for the system of linear equations
    A = []
    B = []

    for i in range(N):
        x, y, z = pts3d[i]
        u, v = pts2d[i]

        # Set up the system of linear equations
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    A = np.array(A)

    # Solve using SVD (Singular Value Decomposition)
    U, S, Vt = np.linalg.svd(A)

    # The last column of Vt is the solution for M (flattened)
    M = Vt[-1].reshape(3, 4)

    return M


def compute_distance(pts2d, pts3d):
    """
    Use find_projection to find matrix M, then use M to compute the average distance
    in the image plane (i.e., pixel locations) between the homogeneous points M X_i and
    2D image coordinates p_i.

    Inputs:
    - pts2d: Numpy array of shape (N, 2), 2D image coordinates
    - pts3d: Numpy array of shape (N, 3), 3D world coordinates

    Returns:
    - distance: Float, average distance between the projected points and 2D points
    """
    # Step 1: Find the projection matrix M
    M = find_projection(pts2d, pts3d)

    # Step 2: Project 3D points into 2D using M
    N = pts3d.shape[0]
    projected_pts2d = []

    for i in range(N):
        X = np.array([pts3d[i][0], pts3d[i][1], pts3d[i][2], 1])  # Homogeneous coordinates
        projected_homogeneous = M @ X
        projected_homogeneous /= projected_homogeneous[2]  # Normalize to homogeneous coordinates
        u_proj, v_proj = projected_homogeneous[0], projected_homogeneous[1]
        projected_pts2d.append([u_proj, v_proj])

    projected_pts2d = np.array(projected_pts2d)

    # Step 3: Calculate the distance between projected points and actual 2D points
    distances = np.linalg.norm(pts2d - projected_pts2d, axis=1)

    # Step 4: Return the average distance
    return np.mean(distances)


if __name__ == '__main__':
    pts2d = np.loadtxt("task1/pts2d.txt")
    pts3d = np.loadtxt("task1/pts3d.txt")

    foundDistance = compute_distance(pts2d, pts3d)
    print("Distance: %f" % foundDistance)
