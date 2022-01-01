BACKEND = 'numpy'

if BACKEND == 'torch':
    from torch import cat, norm, einsum, argmin, argmax, where
    from torch.tensor import Tensor
    expand = Tensor.expand
    dim_arg = 'dim'
    """
    Batches need to be processed independently as the number of points in the hull may differ between them (i.e., the 
    hulls would be a Bx?x2 tensor with varying dim1). Thus, we assume the point tensors to be of shape Nx2.
    """
else:
    from numpy import concatenate as cat
    from numpy import einsum, argmin, argmax, where
    from numpy.linalg import norm
    from numpy import resize as expand
    dim_arg = 'axis'


def distance_from_line(points, point_A, point_B):
    """
    Computes the (signed) distance of all points from the line given by points A and B.
    :param points: 2D points
    :param point_A: start of directional vector
    :param point_B: end of directional vector
    :return: signed distances to AB
    """
    AB = point_B - point_A
    # flip axes and sign of one element -> orthogonal to AB
    n = cat([AB[:, 1][:, None], -AB[:, 0][:, None]], **{dim_arg: 1})
    norm_n = norm(n, **{dim_arg: 1})
    # adapt dimensions
    n = expand(n, (points.shape[0], 2))
    if norm_n == 0:  # if norm_n == 0 -> AB == 0 -> line degenerated to point -> Euclidean distance
        distance_from_AB = norm(points - point_A, **{dim_arg: 1})
    else:  # projection of vector Ap onto n, where p is a query point (batch-wise dot product)
        distance_from_AB = einsum('ij,ij->i', points - point_A, n) / norm_n
    return distance_from_AB


def distance_from_lines(points, lines):
    """
    Computes the (signed) distance of all points from the lines (assumed to be points in ccw order).
    :param points: 2D points
    :param lines: ndarray or tensor of shape Mx2, giving the M points of a polyline in ccw order
    :return: signed distances to each of the lines -> NxM distances
    """
    AB = lines[1:] - lines[:-1]
    # flip axes and sign of one element -> orthogonal to AB
    n = cat([AB[:, 1][:, None], -AB[:, 0][:, None]], **{dim_arg: 1})
    norm_n = norm(n, **{dim_arg: 1})
    # adapt dimensions
    n = expand(n[None, :], (points.shape[0], lines.shape[0] - 1, 2))
    norm_n = expand(norm_n[None, :], (points.shape[0], lines.shape[0]-1))
    # if norm_n == 0 -> AB == 0 -> line degenerated to point -> Euclidean distance
    degenerate = norm_n == 0
    norm_n[degenerate] = 1
    # else: projection of vector Ap onto n, where p is a query point (batch-wise dot product)
    vector_from_AB = points[:, None, :] - lines[:-1][None, ...]
    distance_from_AB = where(degenerate, norm(vector_from_AB, **{dim_arg: 2}),
                             einsum('bij,bij->bi', vector_from_AB, n) / norm_n)
    return distance_from_AB


def convexhull(hull, points, distance_from_AB, point_A, point_B, pos_A=0):
    """
    Recursively computes the partial convex hull for points in the positive half-space defined by the line AB.
    :param hull: list, initialized to [A, B] -- consecutive points are added in-between
    :param points: points for which the convex hull should be computed; assumed to be in 2D
    :param point_A: first extreme point
    :param point_B: second extreme point
    :param pos_A: index of A in hull -- used as pointer to insert points in the right order
    :return: ordered points of the convex hull (positive side)
    """
    if points.shape[0] == 0:
        return 0  # subset empty -> done

    # get farthest point from AB (another extreme point)
    point_C = points[distance_from_AB.argmax()][None, :]
    hull.insert(pos_A + 1, point_C)
    added = 1

    # we get 3 sets: the set S0 in the triangle ACB, S1 right of AC and S2 right of CB
    # -- note: points in S0 can be ignored (inside the hull of ACB and thus inside the hull of the set)
    distance_from_AC = distance_from_line(points, point_A, point_C)
    S1 = distance_from_AC > 0
    added += convexhull(hull, points[S1], distance_from_AC[S1], point_A, point_C, pos_A=pos_A)
    distance_from_CB = distance_from_line(points, point_C, point_B)
    S2 = distance_from_CB > 0
    added += convexhull(hull, points[S2], distance_from_CB[S2], point_C, point_B, pos_A=pos_A + added)
    return added


def quickhull(points):
    """
    Computes the convex hull of a set of 2D points using the QuickHull algorithm.
    :param points: ndarray or tensor of shape Nx2
    :return: subset of the points that constitutes their convex hull in ccw order (thus, outside has positive distance)
    """
    if points.shape[0] == 1:  # single point
        return points

    # get initial extreme points along x and y
    candidates = cat([argmin(points, **{dim_arg: 0}), argmax(points, **{dim_arg: 0})], **{dim_arg: 0})
    candidate_combinations = points[candidates][None, ...] - points[candidates][:, None, :]
    distances = norm(candidate_combinations.reshape(16, 2), **{dim_arg: 1})
    ind_A, ind_B = distances.argmax() % 4, distances.argmax() // 4
    point_A, point_B = points[candidates[ind_A]][None, :], points[candidates[ind_B]][None, :]  # initial points

    # AB divides the points into sets S1 (right of AB) and S2 (right of BA)
    distance_from_AB = distance_from_line(points, point_A, point_B)
    # -- compute hull (recursively for left and right side of AB)
    hull_S1 = [point_A, point_B]
    S1 = distance_from_AB > 0
    convexhull(hull_S1, points[S1], distance_from_AB[S1], point_A, point_B)  # positive side of AB
    hull_S2 = [point_B, point_A]
    S2 = distance_from_AB < 0
    convexhull(hull_S2, points[S2], -distance_from_AB[S2], point_B, point_A)  # negative side of AB
    hull = cat(hull_S1 + hull_S2[1:-1], **{dim_arg: 0})  # note: B and A already in hull of S1
    return hull


def points_in_hull(query_points, hull, tolerance=0):
    """
    Checks whether the query points are inside the convex hull (given by its vertices in ccw order).
    :param query_points: ndarray of shape Nx2
    :param hull: ndarray of shape Mx2, assumed to be in ccw order (outside has positive distance)
    :param tolerance: allow points slighly outside (positive) or reject points only slightly inside (negative value)
    :return: ndarray of shape N with booleans indicating whether the ith point is inside the hull
    """
    if hull.shape[0] == 1:  # single point -- so they must coincide (up to tolerance)
        return norm(query_points - hull, **{dim_arg: 1}) <= tolerance

    # query points are inside if they are on the negative side (i.e., inside) for all hull segments
    segments = cat([hull, hull[0][None, :]], **{dim_arg: 0})  # back to first point to get closed polygon
    distances = distance_from_lines(query_points, segments)  # batch x num points X num segments
    inside = (distances < tolerance).all(**{dim_arg: 1})  # check whether a point is inside all segments
    return inside
