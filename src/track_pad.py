"""@package TrackPad
The container of the point cloud detection result

Author: zukai.wang@cepton.com
Date: 10/18/2022
Version: 1.0.0
"""

import numpy as np
from sklearn.cluster import DBSCAN

from src.target import Target
from src.classifier import get_prob_vec
from src.frame import PointFlag
from src.util import timer_func, get_radius


MAX_OBJECTS = 512

def load_rotation(cfg):
	"""
	Returns a 3x3 rotation matrix corresponding to the given pitch, yaw, and roll angles.
	"""
	# Calculate the sine and cosine of the angles
	sp, cp = np.sin(cfg['pitch']), np.cos(cfg['pitch'])
	sy, cy = np.sin(cfg['yaw']),   np.cos(cfg['yaw'])
	sr, cr = np.sin(cfg['roll']),  np.cos(cfg['roll'])

	# Construct the rotation matrix
	R_pitch = np.array([[ 1,   0,   0], [ 0,  cp, -sp], [  0, sp, cp]])   # along X axis
	R_yaw   = np.array([[cy,   0,  sy], [ 0,   1,   0], [-sy,  0, cy]])   # along Y axis
	R_roll  = np.array([[cr, -sr,   0], [sr,  cr,   0], [  0,  0,  1]])   # along Z axis

	# Combine the three rotations into a single rotation matrix
	return R_yaw @ R_pitch @ R_roll


class Pad():
	"""
	Track pad
	"""
	def __init__(self, cfg, model):
		# Load configuration
		self.cfg = cfg
		self.rotation_matrix = load_rotation(cfg)
		self.lidar_height = cfg["lidar_height"]

		self.t0 = np.uint64(0) # Init time when the track pad starts
		self.timestamp = 0     # The duration of this trackpad running in seconds
		self.frame_dt = 0

		## Gridding Info
		self.n_x_grids = np.uint16(2*cfg["phi_max"] / cfg["d_phi"])
		self.n_y_grids = np.uint16(cfg["dist_max"] / cfg["dist_grid"])
		self.n_z_grids = np.uint16(cfg["n_z_grids"])

		## Grid index edge
		self.ix_min = 0
		self.ix_max = self.n_x_grids-1
		self.iy_min = 0
		self.iy_max = self.n_y_grids-1

		## BKG map: 0 to 65535
		self.bkg_prob = np.zeros((self.n_x_grids, self.n_y_grids, self.n_z_grids), dtype=np.uint16)

		## Ground Surface level (m)
		self.gs_lvl = np.zeros((self.n_x_grids, self.n_y_grids))
		## Ground Surface Sigma Matrix
		self.gs_var = np.zeros((self.n_x_grids, self.n_y_grids)) + cfg["global_dz"]
		self.gs_var[:, :10] = 0

		## DBScan
		self.dbscan = DBSCAN(eps=cfg["eps_cluster"], min_samples=cfg["min_pts_cluster"])

		## Segmentation
		# self.segmenter = DBSCAN(eps=cfg["eps_seg"], min_samples=cfg["min_pts_seg"])

		## Tracking dictionary: (sn, target)
		self.next_sn = np.uint32(1)
		self.track_dict = {}
		self.sn_map = np.zeros((self.n_x_grids, self.n_y_grids), dtype=np.uint32)
		self.ghost_pts = []

		# Classifier
		self.model = model

	def vox_to_pos(self, vox):
		rho = (vox[:, 1] + 0.5) * self.cfg["dist_grid"]
		phi = (vox[:, 0] + 0.5) * self.cfg["d_phi"] - self.cfg["phi_max"]
		return np.c_[rho * np.sin(phi), rho * np.cos(phi), vox[:, 2] * self.cfg["h_grid_size"]].astype(np.float32)
	
	def pos_to_vox(self, pos):
		valid = pos[:, 1] > 0
		if valid.sum() < 1:
			return None
		
		ix = ((np.arctan(pos[valid, 0] / pos[valid, 1]) + self.cfg["phi_max"]) / self.cfg["d_phi"]).astype(np.int16)
		rho = np.sqrt(pos[valid, 0]**2 + pos[valid, 1]**2)
		iy = (rho / self.cfg["dist_grid"]).astype(np.int16)
		ih = (pos[valid, 2] / self.cfg["h_grid_size"]).astype(np.int16)
		ih -= ih.min()

		inside = (ix >=0) & (ix < self.n_x_grids) & (iy >= 0) & (iy < self.n_y_grids)
		if inside.sum() < 1:
			return None
		return np.c_[ix, iy, ih][inside]

	def load_ground_surf(self, surf_pts):
		"""
		Load the ground surface info
		"""
		self.gs_lvl[surf_pts['ix'], surf_pts['iy']] = surf_pts['z'] * 1e-3
		self.gs_var[surf_pts['ix'], surf_pts['iy']] = surf_pts['z_var'] * 1e-4

	def save_ground_surf(self, out_path):
		"""
		Save the ground surface
		"""
		ix, iy = np.where(self.gs_var < 6)
		np.savez(out_path, ix=ix, iy=iy,
						 z=(1e3*self.gs_lvl[ix, iy]).astype(np.int16),
						 z_var=(1e4*self.gs_var[ix, iy]).astype(np.uint16))

	def load_bkg(self, bkg_npz):
		"""
		Load background matrix:
		self.n_x_grids, self.n_y_grids, self.n_z_grids
		"""
		# select within range indices
		within_range  = (bkg_npz['ix'] >= 0) & (bkg_npz['ix'] < self.n_x_grids)
		within_range &= (bkg_npz['iy'] >= 0) & (bkg_npz['iy'] < self.n_y_grids)
		within_range &= (bkg_npz['iz'] >= 0) & (bkg_npz['iz'] < self.n_z_grids)

		self.bkg_prob[bkg_npz['ix'][within_range], bkg_npz['iy'][within_range], bkg_npz['iz'][within_range]] = bkg_npz['prob'][within_range]

	def save_bkg(self, out_path, prob_thresh=2):
		"""
		Save the background matrix
		"""
		ix, iy, iz = np.where(self.bkg_prob > prob_thresh)
		np.savez(out_path, ix=ix.astype(np.uint16), iy=iy.astype(np.uint16), iz=iz.astype(np.uint16),
						 prob=(self.bkg_prob[ix, iy, iz].clip(0, 65535)).astype(np.uint16))

	def map_update(self):
		"""
		Loop through the track_dict:
			1. update the sn_map
			2. Remove the objects that are not to be tracked
			3. update the ghost_pts
			4. clear the indices of each target
		"""
		self.sn_map[:, :] = 0
		self.ghost_pts = []
		for sn in list(self.track_dict.keys()):
			agent = self.track_dict[sn]

			# Anticipate the occupied SN map, the intermediate pixels will be filled up
			if agent.anticipate(self):
				self.track_dict.pop(sn)
				continue

			# convert the vox to pos
			self.ghost_pts.append(self.vox_to_pos(agent.voxels))

			# Clear the points for a new frame association
			self.track_dict[sn].indices = np.array([], dtype=np.uint16)
			self.track_dict[sn].n_pts = 0
			self.track_dict[sn].prob_vec[:] = 0

	def associate_singletons(self, frm):
		s_idx = frm.sig_indices[frm.cid_arr[frm.sig_indices] < 0]
		
		# Loop through the targets, attach the singletons first
		for sn in self.track_dict:
			agent = self.track_dict[sn]
			sel = (self.sn_map[frm.ix[s_idx], frm.iy[s_idx]] == sn) & (frm.hs[s_idx] < self.cfg["max_sig_height"])
			agent.indices = np.hstack([agent.indices, s_idx[sel]])

	@timer_func
	def clustering(self, frm):
		mask = PointFlag.NOISE + PointFlag.GROUND + PointFlag.BKG
		frm.sig_indices = np.where(frm.flag & mask == 0)[0]

		pos_combined = frm.pos[frm.sig_indices]
		if len(self.ghost_pts) > 0:
			self.ghost_pts = np.vstack(self.ghost_pts)
			pos_combined = np.vstack((frm.pos[frm.sig_indices], self.ghost_pts))

		self.dbscan.fit(pos_combined)
		frm.cid_arr[frm.sig_indices] = self.dbscan.labels_[:len(frm.sig_indices)]


	@timer_func
	def tracking(self, frm):
		# Associate the non clustered points to the track list
		self.associate_singletons(frm)

		for c in range(self.dbscan.labels_[:len(frm.sig_indices)].max()+1):
			cluster_idx_arr = frm.sig_indices[self.dbscan.labels_[:len(frm.sig_indices)] == c]

			if len(cluster_idx_arr) < 1:
				# Not enough points
				continue

			h_max = frm.hs[cluster_idx_arr].max()
			# calculate the radius of the cluster
			if h_max > self.cfg["max_sig_height"]:
				# it is a background cluster
				# print(f"height bkg rule: h_max = {h_max:.2f}")
				frm.flag[cluster_idx_arr] |= PointFlag.BKG
				continue

			if h_max < self.cfg["global_dz"]:
				continue

			radius = get_radius(frm.pos[cluster_idx_arr, :2])
			if radius > self.cfg["max_sig_radius"]:
				# it is a background cluster
				# print(f"Radius bkg rule: radius = {radius:.2f}")
				frm.flag[cluster_idx_arr] |= PointFlag.BKG
				continue

			# Calculate the probability array of the cluster
			prob_vec = get_prob_vec(frm.pos[cluster_idx_arr], self)

			# Tracking
			self.target_tracking(frm, cluster_idx_arr, prob_vec)

	def target_register(self, frm, indices, prob_vec):
		if len(indices) < self.cfg["min_sig_pts"]:
			return None

		h_min = frm.hs[indices].min()
		h_max = frm.hs[indices].max()

		if (h_max < self.cfg["min_sig_height"]) | (h_max < h_min + self.cfg["local_dz_max"]) | (h_min > self.cfg["min_sig_height"]):
			return None

		if h_min > self.cfg["global_dz"]:
			prob_vec[:] = 0

		dist = frm.dist_arr[indices].min()

		# Register the target if any of the following requirements are met:
		# 1. Close enough
		# 2. Max probability > 50%
		if (dist < self.cfg["alert_dist"]) | (prob_vec.max() > self.cfg["prob_thresh"]):
			new_tgt = Target(frm, indices, prob_vec, self.next_sn)
			self.track_dict[self.next_sn] = new_tgt
			# Label SN
			frm.sn_arr[indices] = self.next_sn

			self.next_sn += np.uint32(1)

	def target_tracking(self, frm, indices, prob_vec):
		"""
		Return the target that the cluster should be associated:
		one cluster can only associate to 1 SN
		"""
		# xy_grid_indices = np.unique(np.c_[frm.ix[indices], frm.iy[indices]], axis=0)
		sn_set = np.unique(self.sn_map[frm.ix[indices], frm.iy[indices]])
		sn_set = sn_set[sn_set>0]

		if len(sn_set) < 1:
			# Not associated with any tracked targets, might be a new one to append
			self.target_register(frm, indices, prob_vec)
			return None

		max_score, sn_chosen, probs_opt = 0, 0, None
		# Loop through the associated taregt that reaches the largest max probability
		for sn in sn_set:
			# If no overlap in fiducial, just skip
			pts_in_common = (self.sn_map[frm.ix[indices], frm.iy[indices]] == sn).sum()
			if pts_in_common < 1:
				continue

			idx_arr = np.hstack([self.track_dict[sn].indices, indices])
			# print(self.track_dict[sn].indices, indices)

			pos_comb = frm.pos[idx_arr]

			prob_vec_comb = prob_vec.copy()
			if self.track_dict[sn].n_pts > 0:
				prob_vec_comb = get_prob_vec(pos_comb, self)

			if prob_vec_comb.max() < min(0.5, self.track_dict[sn].prob_vec.max()):
				# If attaching this cluster, the probability of it's being an target reduces: pass
				continue

			score = prob_vec_comb.max() * pts_in_common

			if score > max_score:
				sn_chosen, probs_opt = sn, prob_vec_comb
				max_score = score

		if sn_chosen < 1:
			# No association, see if need to append
			self.target_register(frm, indices, prob_vec)
		else:
			# The cluster chose this target[sn_chosen],
			# but the target also gets to choose whether to annex this cluster
			frm.sn_arr[indices] = sn_chosen
			self.track_dict[sn_chosen].associated(indices, probs_opt)


	def tracklist_update(self, frm):
		"""
		Loop through the track_dict:
			1. associate more points 
			2. update the number of the points in current frame
		"""
		for sn in list(self.track_dict.keys()):
			self.track_dict[sn].n_pts = len(self.track_dict[sn].indices)

			# Check if this target can be removed
			if self.track_dict[sn].n_pts < self.cfg["min_pts_cluster"]:
				# This means it has no associated cluster
				if self.timestamp > self.track_dict[sn].t_last + self.cfg["untracked_time_cut"]:
					self.track_dict.pop(sn)

			else:
				# update the trajectory
				self.track_dict[sn].tracked_update(frm, self)


	def static_check(self):
		"""
		Loop through the track_dict to check every moving target
		and mark the voxels of the bkg probability to 0
		"""
		for sn in self.track_dict:
			tgt = self.track_dict[sn]
			if tgt.is_dynamic:
				self.bkg_prob[tgt.voxels[:, 0], tgt.voxels[:, 1], tgt.voxels[:, 2]]\
					  -= self.bkg_prob[tgt.voxels[:, 0], tgt.voxels[:, 1], tgt.voxels[:, 2]].clip(0, self.cfg["sig_rate"])
			elif tgt.is_static:
				self.bkg_prob[tgt.voxels[:, 0], tgt.voxels[:, 1], tgt.voxels[:, 2]] = self.bkg_prob[tgt.voxels[:, 0], tgt.voxels[:, 1], tgt.voxels[:, 2]].clip(0, 65553-self.cfg["max_delta_bkg_rate"])
				self.bkg_prob[tgt.voxels[:, 0], tgt.voxels[:, 1], tgt.voxels[:, 2]] += self.cfg["max_delta_bkg_rate"]


	@timer_func
	def bkg_update(self, frm):
		"""
		Update the background matrix
		"""
		is_bkg = frm.flag & PointFlag.BKG > 0
		frm_vox = np.c_[frm.ix, frm.iy, frm.ih]

		# Update background probability
		inside_range  = (frm.ih >= 0) & (frm.ih < self.n_z_grids) & (frm.flag & PointFlag.GROUND == 0)
		inside_range &= (frm.ix >= 0) & (frm.ix < self.n_x_grids)
		inside_range &= (frm.iy >= 0) & (frm.iy < self.n_y_grids)
		all_vox = np.unique(frm_vox[inside_range], axis=0)
		bkg_vox = np.unique(frm_vox[inside_range & is_bkg], axis=0)

		max_prob = 65530 - 27*(self.cfg["bkg_rate"] + self.cfg["max_delta_bkg_rate"])
		self.bkg_prob = self.bkg_prob.clip(self.cfg["sig_rate"], max_prob)

		for dx in range(-1, 2):
			ix_all = (all_vox[:, 0] + dx).clip(0, self.n_x_grids-1)
			ix_bkg = (bkg_vox[:, 0] + dx).clip(0, self.n_x_grids-1)
			for dy in range(-1, 2):
				iy_all = (all_vox[:, 1] + dy).clip(0, self.n_y_grids-1)
				iy_bkg = (bkg_vox[:, 1] + dy).clip(0, self.n_y_grids-1)

				for dz in range(-1, 2):
					iz_all = (all_vox[:, 2] + dz).clip(0, self.n_z_grids-1)
					iz_bkg = (bkg_vox[:, 2] + dz).clip(0, self.n_z_grids-1)

					self.bkg_prob[ix_all, iy_all, iz_all] += self.cfg["bkg_rate"]
					self.bkg_prob[ix_bkg, iy_bkg, iz_bkg] += self.cfg["max_delta_bkg_rate"]

		# bkg 2D map
		# base_sel = bkg_vox[:, 2] < self.cfg["base_ground_grid_cut"]
		# ixy_bkg = np.unique(bkg_vox[base_sel, :2], axis=0)
		# self.bkg_prob[ixy_bkg[:, 0], ixy_bkg[:, 1], :5] += self.cfg["max_delta_bkg_rate"]
		self.bkg_prob -= self.cfg["sig_rate"]


	def frame_processing(self, frm):
		self.frame_dt = frm.ts[0] - self.timestamp
		self.timestamp = frm.ts[0]
		frm.gridding(self)
		frm.ground_label(self)
		frm.height_gridding(self)
		frm.bkg_label(self)
		# frm.unmark_grd()

		self.map_update()

		self.clustering(frm)
		self.tracking(frm)

		self.tracklist_update(frm)

		# self.static_check()

		# frm.update_ground_surf(self)
		# self.bkg_update(frm)
