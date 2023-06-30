import numpy as np
from src.bbox import get_bbox
from src.util import get_radius


class ClusterInfo():
	"""
	cluster info
	"""
	def __init__(self, in_a, in_b) -> None:
		## indices array of points inside the cluster
		self.in_a = in_a
		self.in_b = in_b

		self.n_pts_a = len(in_a)
		self.n_pts_b = len(in_b)
		self.n_pts = self.n_pts_a + self.n_pts_b

		## Matched points in previous frame
		self.matched_a = 0

		## Number of points within the fiducial volumn
		self.matched_b = 0

		## matching score
		self.matched_score = 0

		## Radius of the cluster
		self.radius = 0

		## Bounding box of this cluster
		self.vertices = np.zeros((4,2))
		self.width = self.length = 0

		## Probability Array
		self.prob_vector = np.zeros(16)

		## Associated target: the entire cluster belongs to the target
		## Association requirements:
		# 1. >=50% of in_a points are assigned to the target in frame a
		# 2. >=50% of in_b points are within anticipated fiducial box
		self.associated_target = None

		## List of affiliated targets
		self.affiliated_list = []

		## Category: 1 => pedestrian or cyclist; 2 => vehicles
		self.category = 0

		## assignment indicator:
		# 0: ommit this cluster, don't do anything
		# 1: append the entire cluster as a new object
		# 2: segment the cluster into sub clusters and append the new sub clusters
		# 3: assign the entire cluster to an existing target
		# 4: assign the cluster to k affiliated targets
		self.assign_indicator = 0

		# Allocate the points
		self.xy_arr, self.h_arr = None, None

	def construct_h_arr(self, frm_a, frm_b):
		"""
		Calculate the height of the cluster
		"""
		self.h_arr = np.hstack([frm_a.hs[self.in_a], frm_b.hs[self.in_b]])

	def set_radius(self, frm_a, frm_b):
		"""
		Calculate the radius
		"""
		self.xy_arr = np.vstack([frm_a.pos[self.in_a, :2], frm_b.pos[self.in_b, :2]])
		self.radius = get_radius(self.xy_arr)

	def bkg_check(self, frm_a, frm_b, cfg):
		"""
		Return the bool whether this cluster is a background cluster:
		if not background, it will set the radius and points for the cluster
		"""
		self.construct_h_arr(frm_a, frm_b)
		h_min, h_max = self.h_arr.min(), self.h_arr.max()
		if h_max > min(h_min, cfg["global_dz"]) + cfg["height_cut"] - cfg["h_grid_size"]:
			# the top point is too high
			return True

		if h_max < h_min + cfg["global_dz"]:
			# the height is too low
			return True

		if h_min > cfg["lidar_height"]:
			# the cluster is too high above
			return True

		self.set_radius(frm_a, frm_b)

		return self.radius > cfg["max_sig_radius"]

	def matcher(self, frm_a, frm_b, target):
		"""
		Set the matched_a and matched_b, which are used to
		determine whether this cluster needs to be segmented
		"""
		matched_a = (frm_a.sn_arr[self.in_a] == target.sn).sum()
		matched_b = 0
		if len(self.in_b) > 0:
			# Fetch the points in frame b
			xy_b = frm_b.pos[self.in_b, :2]
			# Append the center position
			xy_b = np.vstack([xy_b, xy_b.mean(axis=0)])
			matched_b = (target.in_fiducial(xy_b)).sum()

		score = (matched_a + 1) * matched_b
		if score > self.matched_score:
			self.matched_score = score
			self.matched_a, self.matched_b = matched_a, matched_b

		return matched_a, matched_b

	def set_prob_vector(self, model, cfg):
		"""
		Set the prob vector for the cluster
		"""
		points = np.c_[self.xy_arr, self.h_arr]

		# Set bounding box
		bbox = get_bbox(points[:, :2])
		self.vertices = bbox.vertices
		self.width, self.length = bbox.width, bbox.length

		# convert the mm to m
		self.prob_vector[:len(model['mu'])] = get_prob_arr(points, model, cfg)

		# set the category
		if self.prob_vector[:2].sum() > 0.5:
			# Pedestrian or Cyclist
			self.category = 1
		elif self.prob_vector[2:].sum() > 0.5:
			self.category = 2

	def sn_label(self, cid, model, cfg):
		"""
		Label the points in the cluster with proper SN id
		"""
		if (len(self.affiliated_list) == 1) & (
			not self.associated_target
		) and self.affiliated_list[0].cid == cid:
			self.associated_target = self.affiliated_list[0]

		if self.associated_target:
			self.assign_indicator = 3
			return None

		if len(self.affiliated_list)==0:
			# No affiliated targets
			# Calculate the probability of this cluster
			self.set_prob_vector(model, cfg)
			if self.prob_vector.sum() > 0.04:
				# Append the new object
				self.assign_indicator = 1
			elif (self.n_pts > cfg["min_segment_npts"]) & (self.n_pts > cfg["min_segment_density"] * self.radius**2):
				# Segement this cluster into smaller sub clusters and append
				self.assign_indicator = 2
		elif len(self.affiliated_list) > 1:
			self.assign_indicator = 4
