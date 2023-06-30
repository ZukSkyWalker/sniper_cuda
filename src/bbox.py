import numpy as np

HALF_PI = 0.5 * np.pi
QUATER_PI = 0.25 * np.pi


class BoundingBox():
	"""
	Finding bounding box based on convex hull
	"""
	def __init__(self, points, heading_init, eps_x=0.1) -> None:
		self.points = points
		self.eps_x = eps_x
		self.edge_weight = 0
		self.score = 0

		# Set default box
		x0, x1 = points[:, 0].min(), points[:, 0].max()
		y0, y1 = points[:, 1].min(), points[:, 1].max()
		self.width = x1 - x0
		self.length = y1 - y0

		self.area = self.width * self.length
		
		self.vertices = np.array([[x0, y0],[x1, y0],[x1, y1],[x0, y1]])
		self.heading = heading_init

	def get_vertices(self, heading):
		"""
		Return the bool array of the points with outlier filtered
		"""
		if heading > np.pi:
			heading -= 2 * np.pi
		elif heading < -np.pi:
			heading += 2 * np.pi

		cos_val, sin_val = np.cos(heading), np.sin(heading)

		# Rotate the coordinates to calculate the bounding box
		pos = self.points.dot(np.array([[cos_val, sin_val], [-sin_val, cos_val]]))

		x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
		x0, x1 = x_min, x_max

		in_x_range = (pos[:, 0] > x_min) & (pos[:, 0] < x_max)
		if in_x_range.sum() > 0:
			x0 = pos[in_x_range, 0].min()
			x1 = pos[in_x_range, 0].max()
		
		y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
		y0, y1 = y_min, y_max

		in_y_range = (pos[:, 1] > y_min) & (pos[:, 1] < y_max)
		if in_y_range.sum() > 0:
			y0 = pos[in_y_range, 1].min()
			y1 = pos[in_y_range, 1].max()

		# count effective edge counts
		rear = pos[in_y_range, 1] < y0 + self.eps_x
		rear_cnt = 0
		if rear.sum() > 0:
			side_len = pos[in_y_range, 0][rear].max() - pos[in_y_range, 0][rear].min()
			rear_cnt = min(rear.sum(), side_len/self.eps_x) * side_len

		front = pos[in_y_range, 1] > y1 - self.eps_x
		front_cnt = 0
		if front.sum() > 0:
			side_len = pos[in_y_range, 0][front].max() - pos[in_y_range, 0][front].min()
			front_cnt = min(front.sum(), side_len/self.eps_x) * side_len

		on_left = pos[in_x_range, 0] < x0 + self.eps_x
		left_cnt = 0
		if on_left.sum() > 0:
			side_len = pos[in_x_range, 1][on_left].max() - pos[in_x_range, 1][on_left].min()
			left_cnt = min(on_left.sum(), side_len/self.eps_x) * side_len

		on_right = pos[in_x_range, 0] > x1 - self.eps_x
		right_cnt = 0
		if on_right.sum() > 0:
			side_len = pos[in_x_range, 1][on_right].max() - pos[in_x_range, 1][on_right].min()
			right_cnt = min(on_right.sum(), side_len/self.eps_x) * side_len

		# print(f"Heading: {heading:.2f}", rear_cnt, left_cnt, right_cnt)

		# Set vertices
		## 3|2
		## 0|1
		vertices = np.array([[x0, y0],[x1, y0],[x1, y1],[x0, y1]]) @ np.array([[cos_val, -sin_val],  [sin_val, cos_val]])

		return vertices, x1-x0, y1-y0, [rear_cnt, front_cnt, left_cnt, right_cnt]


	def edge_check(self, heading):
		"""
		Return the stop flag: if true, no further edge check necessary
		"""
		vertices, delta_x, delta_y, cnt_arr = self.get_vertices(heading)
		# You will see two sides, pick the largest combination
		edge_w = max(cnt_arr[:2]) + max(cnt_arr[2:])
		score = edge_w / ((delta_x+0.1) * (delta_y+0.1))

		# print(f"theta = {heading:.2f}, score = {score:.2f}")

		# Compare edge points density
		if score > self.score:
			# print(f"heading={heading}, edge_weight={edge_w}")
		# if delta_x * delta_y < self.delta_x * self.delta_y:
			self.width = delta_x
			self.length = delta_y

			self.vertices = vertices
			self.heading = heading
			self.edge_weight = edge_w
			self.score = score

	def set_box(self, d_theta=0.01):
		"""
		Brute force: loop through all the possible angles
		"""
		heading_0 = self.heading
		for theta in np.arange(heading_0-np.pi*0.25, heading_0 + np.pi*0.25, d_theta):
			self.edge_check(theta)

def get_bbox(points, heading_init=0, d_theta=0.01, eps_x=0.1):
	"""
	Get the bounding box info
	"""
	bbox = BoundingBox(points, heading_init, eps_x=eps_x)
	bbox.set_box(d_theta)

	return bbox

def get_bbox_with_direction(points, heading):
	"""
	Given the determined heading, return the bounding box info
	"""
	bbox = BoundingBox(points, heading)
	vertices, width, length, cnt_arr = bbox.get_vertices(heading)

	return vertices, width, length, sum(cnt_arr)
	