
from time import time
import numpy as np
from scipy.spatial import ConvexHull, QhullError


def timer_func(func):
	# This function shows the execution time of
	# the function object passed
	def wrap_func(*args, **kwargs):
		t1 = time()
		result = func(*args, **kwargs)
		t2 = time()
		print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
		return result
	return wrap_func

def get_radius(xy_arr):
	# get center
	xc = 0.5*(xy_arr[:, 0].min() + xy_arr[:, 0].max())
	yc = 0.5*(xy_arr[:, 1].min() + xy_arr[:, 1].max())
	return np.sqrt((xy_arr[:, 0] - xc)**2 + (xy_arr[:, 1]-yc)**2).max()


def plane_fit(pos, min_resolution=1e-4):
	"""
	 Return the ground plane parameters
	 O(N) complexity
	 If the plane fit result is not valid, return a default flat plane
	"""
	plane_par = np.array([0, 0, pos[:, 2].min()])
	if len(pos) < 4:
		return plane_par
	x0, y0, z0 = (pos).mean(axis=0)

	dx = pos[:, 0] - x0
	dy = pos[:, 1] - y0
	dz = pos[:, 2] - z0

	pxx = (dx*dx).sum()
	pyy = (dy*dy).sum()
	pxy = (dx*dy).sum()

	pxz = (dx*dz).sum()
	pyz = (dy*dz).sum()

	# Calculate the slope
	dxy = pxx * pyy - pxy * pxy
	if np.abs(dxy) < min_resolution:
		return plane_par

	a = (pxz * pyy - pyz * pxy) / dxy
	b = (pxx * pyz - pxz * pxy) / dxy

	return np.array([a, b, (z0 - a*x0 - b*y0)])


def in_angular_range(points, p1, p2):
	"""
	The p1 abd p2 are two neighboring vertices in the counter clockwise sequence
	Return the bool array of representing which points within the angular range
	"""
	# p1 at 1st quadrant
	if (p1[0] >= 0) & (p1[1] >= 0):
		sel = (points[:, 1] >= 0) & (points[:, 0]* p1[1] <= points[:, 1] * p1[0])
		sel &= (points[:, 1] * p2[0] <= points[:, 0] * p2[1])

		# p2 at 3rd quadrant
		if p2[1] < 0:
			sel |= (points[:, 0] < 0) & (points[:, 1] < 0) & (points[:, 0] * p2[1] >= points[:, 1] * p2[0])
		return sel

	# p1 at 2nd quadrant
	if (p1[0] <= 0) & (p1[1] >= 0):
		sel = (points[:, 0] <= 0) & (points[:, 1] >=0) & (points[:, 0]* p1[1] <= p1[0]*points[:, 1])
		
		if (p2[0] <= 0) & (p2[1] >= 0):
			sel &= (points[:, 0]* p2[1] >= p2[0]*points[:, 1])
			return sel
		if (p2[0] <= 0) & (p2[1] <= 0):
			sel |= (points[:, 0] <= 0) & (points[:, 1] < 0) & (points[:, 0] * p2[1] >= points[:, 1] * p2[0])
			return sel
		
		# p2 in 4th quadrant
		sel |= (points[:, 0] <= 0) & (points[:, 1] <= 0)
		sel |= (points[:, 0] >= 0) & (points[:, 1] <= 0) & (points[:, 0] * p2[1] >= points[:, 1] * p2[0])

		return sel

	# p1 at third quadrant
	if (p1[0] <= 0) & (p1[1] <= 0):
		# counter clockwise to p1
		sel = (points[:, 0] <= 0) & (points[:, 1] <= 0) & (points[:, 0] * p1[1] <= points[:, 1] * p1[0])

		# p2 at third quadrant
		if (p2[0] <= 0) & (p2[1] <= 0):
			sel &= points[:, 0] * p2[1] >= points[:, 1] * p2[0]
			return sel
		
		# p2 at 4th quadrant
		if (p2[0] >= 0) & (p2[1] < 0):
			# print("4th quadrant p2")
			sel |= (points[:, 0] >= 0) & (points[:, 1] < 0) & (points[:, 1] * p2[0] < points[:, 0] * p2[1])
			return sel
		
		# p2 at 1st quadrant
		sel |= (points[:, 0] >= 0) & (points[:, 1] <= 0)
		sel |= (points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] * p2[1] > points[:, 1] * p2[0])
		return sel

	# p1 at 4th quadrant
	sel = (points[:, 0] > 0) & (points[:, 1] < 0) & (points[:, 0] * p1[1] <= points[:, 1] * p1[0])
	# p2 at 4th quadrant
	if (p2[0] >= 0) & (p2[1] <= 0):
		sel &= points[:, 0] * p2[1] >= points[:, 1] * p2[0]
		return sel
	# # p2 at 1st quadrant
	if (p2[0] >= 0) & (p2[1] >= 0):
		sel |= (points[:, 0] >= 0) & (points[:, 1] >= 0) & (points[:, 0] * p2[1] > points[:, 1] * p2[0])
		return sel
	# # p2 at 2nd quadrant
	sel |= (points[:, 0] >= 0) & (points[:, 1] >= 0)
	sel |= (points[:, 0] <= 0) & (points[:, 1] >= 0) & (points[:, 0] * p2[1] > points[:, 1] * p2[0])
	return sel


def intersection_dist(points, p1, p2):
	"""
	Return the directional distance from the points to their corresponding intersections:
	points to center distance - intersection to center distance
	"""
	points_distances = np.sqrt((points**2).sum(axis=1))

	x1, y1 = p1
	x2, y2 = p2

	denom = points[:, 0] * (y2 - y1) - points[:, 1] * (x2 - x1)
	factor = (x1 * y2 - x2 * y1) / denom

	# Calculate the intersection distances
	xi = points[:, 0] * factor
	yi = points[:, 1] * factor

	return points_distances / np.sqrt(xi**2 + yi**2) - 1


def distance_2_box(points, box):
	"""
	Both inputs need to be shifted so the box center is (0, 0)
	input:
		* points: 2D numpy array (x,y)
		* edge: (x1, y1) to (x2, y2)

	output:
		Distance to the box (if the point is within the box, return a negative value)
	"""
	# Shift the origin:
	box_center = box.mean(axis=0)
	points -= box_center
	box -= box_center

	dist_arr = np.zeros(len(points))

	for v_idx in range(4):
		p1, p2 = box[v_idx], box[(v_idx+1)%4]
		sel = in_angular_range(points, p1, p2)
		dist_arr[sel] = intersection_dist(points[sel], p1, p2)

	return dist_arr

def get_convex_vertices(xy):
	try:
		hull = ConvexHull(xy)
		return hull.points[hull.vertices].astype(np.float32)
	except QhullError:
		return None