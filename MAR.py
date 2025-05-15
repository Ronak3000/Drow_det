from scipy.spatial import distance as dist

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[6])
    B = dist.euclidean(mouth[3], mouth[5])
    C = dist.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)
