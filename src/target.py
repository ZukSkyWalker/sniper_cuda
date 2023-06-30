import numpy as np
from src.frame import PointFlag
from src.classifier import get_prob_vec


CATEGORIES = ["vehicle", "pedestrian", "cyclist"]

TRACK_FRAMES = 8

class Target():
	"""
	Object in the track list
	"""
	def __init__(self, frm, indices, prob_vec, sn):
		self.sn = sn
		# Get index array
		self.voxels = np.unique(np.c_[frm.ix[indices], frm.iy[indices], frm.ih[indices]], axis=0)
		
		# Down sampled positions array to be calculated based on voxels
		self.pos = None

		# Initialize the probabiolity array
		self.prob_vec = prob_vec
		self.indices = indices
		self.n_pts = len(indices)

		## First seen space & time
		self.t_init = frm.ts[indices].mean()
		self.pos_init = frm.pos[indices].mean(axis=0)
		self.max_dist_away = 0

		## Last seen time (avg time of the points)
		self.t_last   = self.t_init.copy()
		self.pos_last = self.pos_init.copy()

		## Allocate the trajectory: x, y, t, w
		self.trajectory = np.zeros((TRACK_FRAMES, 4))
		self.trj_idx    = 0
		# self.trajectory[0][:2] = self.pos_init[:2].copy()
		# self.trajectory[0][2]  = self.t_init
		# self.trajectory[0][3]  = self.n_pts

		## Set velocity
		self.velocity = np.zeros(2)

		## Whether the object is static
		self.is_dynamic = False
		self.is_static = False

		## set type string
		self.category = "unknown"
		self.category = self.get_category()

		## Mark the flag
		if self.category == "vehicle":
			frm.flag[indices] = PointFlag.VEHCLE
		elif self.category == "pedestrian":
			frm.flag[indices] = PointFlag.PED
		elif self.category == "cyclist":
			frm.flag[indices] = PointFlag.CYCLIST

		## Mark the sn
		frm.sn_arr[indices] = sn

	def set_velocity(self):
		"""
		Use the trajectory buffer to calculate the velocity (m/s)
		"""
		# Select the trajectories with points > 0
		valid = self.trajectory[:, -1] > 0

		if valid.sum() < 2:
			# Stop here
			return None

		# Perform linear fit
		t_arr = (self.trajectory[valid, 2] - self.trajectory[valid, 2].mean())

		# use the number of the points as the weight
		w_arr = self.trajectory[valid, -1]
		# print(f"traject of {self.sn}:", self.trajectory[valid])

		for axis in [0, 1]:
			x_arr = (self.trajectory[valid, axis] - self.trajectory[valid, axis].mean())
			# print(x_arr, t_arr, w_arr)
			self.velocity[axis] = (x_arr * t_arr * w_arr).sum() / (w_arr * t_arr**2).sum()

	def get_category(self):
		idx = self.prob_vec.argmax()

		# If this target has previously non-unknown category
		if (self.prob_vec[idx] < 0.5) & (self.category != CATEGORIES[idx]):
			return "unknown"

		return CATEGORIES[idx]
	
	def associated(self, indices, prob_vec):
		self.indices = np.hstack([indices, self.indices])
		self.prob_vec = prob_vec.copy()

	def tracked_update(self, frm, t_pad):
		"""
		1. Update the voxel
		2. Update the prob_vec
		3. Update the velocity
		4. Update the trajectory
		"""

		# points in the current frame
		pos_arr_frm = frm.pos[self.indices]

		if self.prob_vec.max() < 0.5:
			pos_cached = t_pad.vox_to_pos(self.voxels)

			pos_comb = np.vstack([pos_arr_frm, pos_cached])
			prob_vec_comb = get_prob_vec(pos_comb, t_pad)

			if prob_vec_comb.max() > self.prob_vec.max():
				self.prob_vec = prob_vec_comb.copy()
				# combine the cached vox
				vox_comb = t_pad.pos_to_vox(pos_comb)
				self.voxels = np.unique(vox_comb, axis=0)
			else:
				self.voxels = np.unique(t_pad.pos_to_vox(pos_arr_frm), axis=0)
		else:
			self.voxels = np.unique(t_pad.pos_to_vox(pos_arr_frm), axis=0)

		# Update the trajectory
		self.pos_last = pos_arr_frm.mean(axis=0).copy()
		self.t_last = frm.ts[self.indices].mean()
		self.trajectory[self.trj_idx,:2] = self.pos_last[:2].copy()
		self.trajectory[self.trj_idx, 2] = self.t_last
		self.trajectory[self.trj_idx, 3] = self.n_pts

		self.trj_idx = (self.trj_idx + 1) % TRACK_FRAMES

		# print("Agent update: ", self.n_pts, self.trajectory)

		if self.is_static & (self.category == 'unknown'):
			frm.flag[self.indices] |= PointFlag.BKG
			# No need to continue
			return None

		# update velocity
		self.set_velocity()

		# Update the category
		self.category = self.get_category()

		# label the frame flag
		if self.category == "vehicle":
			frm.flag[self.indices] = PointFlag.VEHCLE
		elif self.category == "pedestrian":
			frm.flag[self.indices] = PointFlag.PED
		elif self.category == "cyclist":
			frm.flag[self.indices] = PointFlag.CYCLIST


	def anticipate(self, t_pad):
		"""
		Every target in the tracked dictionary
		Update the voxels according to the velocity

		Return whether this agent needs to be removed
		"""
		# 1. Check if the target has not been tracked over time
		
		self.n_pts = 0
		# 2. convert the voxels to positions
		pos = t_pad.vox_to_pos(self.voxels)

		# 3. shift by velocty * dt
		pos[:, :2] += self.velocity * t_pad.frame_dt

		# 4. convert back to grids
		self.voxels = t_pad.pos_to_vox(pos)

		# label SN map
		if not (self.voxels is None):
			self.label_sn_map(t_pad)
			return False
		
		return True


	def label_sn_map(self, t_pad):
		"""
		Fill up the SN map
		"""
		for iy in np.unique(self.voxels[:, 1]):
			in_y = self.voxels[:, 1] == iy
			ix0 = self.voxels[in_y, 0].min()
			ix1 = self.voxels[in_y, 0].max()+1
			t_pad.sn_map[ix0:ix1, iy] = self.sn


	def moving_check(self, t_pad):
		"""
		Every target will need to be checked, update the trajectory index here
		Also update the trj_idx
		"""
		# Update trajectory index
		self.trj_idx = (self.trj_idx + 1) % TRACK_FRAMES

		# Update the max_dist_away from the first discovered place
		new_dist = np.sqrt(((self.pos_last[:2] - self.pos_init[:2])**2).sum())
		self.max_dist_away = max(self.max_dist_away, new_dist)
		
		if (self.max_dist_away > t_pad.cfg["shift_thresh"]) & ((self.velocity**2).sum() > t_pad.cfg["v_var_thresh"]):
			self.is_dynamic = self.n_pts > 0
		elif self.t_last > self.t_init + t_pad.cfg["static_time_threshold"]:
			self.is_static = True
			self.is_dynamic = False
		else:
			self.is_dynamic = False
