import numpy as np, math, cv2

_model_points = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
])

def _isRotationMatrix(R):
    return np.linalg.norm(np.eye(3) - R.T.dot(R)) < 1e-6

def _rotationMatrixToEulerAngles(R):
    assert _isRotationMatrix(R)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy < 1e-6:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    else:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    return np.array([x, y, z])

def getHeadTilt(image_size, image_points):
    h, w = image_size
    focal = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal, 0,     center[0]],
        [0,     focal, center[1]],
        [0,     0,     1       ]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))
    _, rvec, _ = cv2.solvePnP(
        _model_points, image_points, camera_matrix,
        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    R, _ = cv2.Rodrigues(rvec)
    tilt = abs(-180 - np.degrees(_rotationMatrixToEulerAngles(R))[0])
    return tilt
