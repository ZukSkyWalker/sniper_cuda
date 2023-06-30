"""@package single_frame_reco
This module takes in a full frame with full coverage of the lidar view,
returns the flag of each point for later processing.
Appreximation is made based on the assumption that the frame length is in the order
of 0.1s so the scene is roughly static.

Author: zukai.wang@cepton.com
Date: 10/18/2022
Version: 1.0.0
"""
import numpy as np
from enum import IntFlag
import plotly.graph_objects as go

from src.util import timer_func


class PointFlag(IntFlag):
	GROUND  = 1
	BKG     = 2
	VEHCLE  = 4
	PED     = 8
	CYCLIST = 16
	NOISE   = 64

class Frame():
	"""
		Information of the point cloud of the corresponding frame:
	"""
	def __init__(self, sdk_frm, tpad):
		valid = ~sdk_frm.invalid
		# Initialize the time & reflectivity
		self.ts = (sdk_frm.timestamps[valid] - tpad.t0) * 1e-6
		self.rs = sdk_frm.reflectivities[valid]

		# Positions
		self.pos = sdk_frm.positions[valid] @ tpad.rotation_matrix
		self.pos[:, 2] += tpad.lidar_height
		self.pos = self.pos.astype(np.float32)

		# Heights array
		self.hs = self.pos[:, 2].copy()

		# Number of the points
		self.n_points = len(self.hs)

		# Flag Array
		self.flag = np.zeros(self.n_points, dtype=np.uint8)

		## SN array: for debugging
		self.sn_arr = np.zeros(self.n_points, dtype=np.uint32)
		self.cid_arr = np.zeros(self.n_points, dtype=np.int32) - 1

		## Reserve index array
		self.ix = None
		self.iy = None
		self.ih = None

		## Initialize the probability of the points being background
		self.bkg_probs = np.zeros(self.n_points, dtype=np.uint16)

		## 2D distances of points:
		self.dist_arr = np.sqrt(self.pos[:, 0]**2 + self.pos[:, 1]**2)
		self.z_buff = (self.dist_arr * tpad.cfg["global_slope_cut"]) + tpad.cfg["local_dz_max"]


	def gridding(self, t_pad):
		"""
		Angular gridding
		"""
		self.ix = ((np.arctan(self.pos[:, 0] / self.pos[:, 1]) + t_pad.cfg["phi_max"]) / t_pad.cfg["d_phi"]).astype(np.int16)
		self.iy = (self.dist_arr / t_pad.cfg["dist_grid"]).astype(np.int16)


	def ground_label(self, t_pad):
		"""
		Return a 2D matrix storing the min z value in each grid
		"""
		# label outside indices
		out_of_bound  = (self.ix < 0) | (self.ix >= t_pad.n_x_grids)
		out_of_bound |= (self.iy < 0) | (self.iy >= t_pad.n_y_grids)
		self.flag[out_of_bound] = PointFlag.NOISE

		idx_arr = np.where(~out_of_bound)[0]

		# Backup height array
		self.hs[idx_arr] = self.pos[idx_arr, 2] - t_pad.gs_lvl[self.ix[idx_arr], self.iy[idx_arr]]

		# Global Z cut
		below_grd = self.pos[idx_arr, 2] < -self.z_buff[idx_arr]
		abv_grd = self.pos[idx_arr, 2] > self.z_buff[idx_arr]

		gs_std = np.sqrt(t_pad.gs_var[self.ix[idx_arr], self.iy[idx_arr]])

		# below_grd = self.hs[idx_arr] < -self.z_buff[idx_arr]
		# below_grd |= self.hs[idx_arr] < -gs_std.clip(self.z_buff[idx_arr], t_pad.cfg["global_dz"])
		abv_grd |= self.hs[idx_arr] > gs_std.clip(t_pad.cfg["local_dz_min"], t_pad.cfg["local_dz_max"])
		self.flag[idx_arr[below_grd]] = PointFlag.NOISE

		grd_idx = idx_arr[(~(below_grd | abv_grd)) & (self.dist_arr > t_pad.cfg["min_grd_dist"]) & (self.dist_arr < t_pad.cfg["max_grd_dist"])]
		self.flag[grd_idx] = PointFlag.GROUND
	
	def get_ground_surface(self, t_pad):
		# Initialize the ground surface high enough
		gs_mat = np.ones(t_pad.gs_lvl.shape) + t_pad.cfg["global_dz"]

		# Cover the neighboring grids
		ix0 = np.maximum(0, self.ix - t_pad.cfg["nb_grids_x"])
		ix1 = np.minimum(self.ix + t_pad.cfg["nb_grids_x"]+1, t_pad.n_x_grids)
		iy0 = np.maximum(0, self.iy - t_pad.cfg["nb_grids_y"])
		iy1 = np.minimum(self.iy + t_pad.cfg["nb_grids_y"]+1, t_pad.n_y_grids)
		
		for i in np.where(self.flag & PointFlag.GROUND > 0)[0]:
			gs_mat[ix0[i]:ix1[i], iy0[i]:iy1[i]] = gs_mat[ix0[i]:ix1[i], iy0[i]:iy1[i]].clip(-99, self.pos[i, 2])

		return gs_mat


	@timer_func
	def update_ground_surf(self, t_pad):
		"""
		Update the ground surface in t_pad
		"""
		gs_mat = self.get_ground_surface(t_pad)
		measured = gs_mat < t_pad.cfg["global_dz"]

		# Update the surface
		ix_lo, iy_lo = np.where(measured & (gs_mat <= t_pad.gs_lvl))
		kg_lo = t_pad.gs_var[ix_lo, iy_lo] / (t_pad.gs_var[ix_lo, iy_lo] + t_pad.cfg["local_dz_max"]**2)
		t_pad.gs_lvl[ix_lo, iy_lo] += kg_lo * (gs_mat[ix_lo, iy_lo] - t_pad.gs_lvl[ix_lo, iy_lo])

		ix_up, iy_up = np.where(measured & (gs_mat > t_pad.gs_lvl))
		kg_hi = 0.01 # slow recover to higher surface
		t_pad.gs_lvl[ix_up, iy_up] += kg_hi * (gs_mat[ix_up, iy_up] - t_pad.gs_lvl[ix_up, iy_up])


	@timer_func
	def bkg_label(self, t_pad):
		"""
		Label all the points with bkg indices
		"""
		idx_arr = np.where((self.flag & (PointFlag.NOISE + PointFlag.GROUND) == 0))[0]

		for i in idx_arr:
			in_range  = (self.ix[i] > -1) & (self.ix[i] < t_pad.bkg_prob.shape[0])
			in_range &= (self.iy[i] > -1) & (self.iy[i] < t_pad.bkg_prob.shape[1])
			in_range &= (self.ih[i] > -1) & (self.ih[i] < t_pad.bkg_prob.shape[2])

			if in_range:
				self.bkg_probs[i] = t_pad.bkg_prob[self.ix[i], self.iy[i], self.ih[i]]

		self.flag[(self.bkg_probs >= t_pad.cfg["bkg_prob_thresh"])] |= PointFlag.BKG

	def unmark_grd_slow(self, indices):
		"""
		In this function, get the points where connect to the base of the cluster
		1. unmark the ground flag
		2. append the indices to the cluster
		"""
		vox = np.unique(np.c_[self.ix[indices], self.iy[indices], self.ih[indices]], axis=0)

		for gidx in vox:
			new_indices = (self.ix == gidx[0]) & (self.iy == gidx[1]) & (self.ih > gidx[2]-3) & (self.ih < gidx[2])
			self.flag[new_indices] &= (255 - PointFlag.GROUND)
			indices = np.vstack((indices, new_indices))

	@timer_func
	def unmark_grd(self):
		"""
		In this function, get the points where connect to the base of the cluster
		1. unmark the ground flag
		2. append the indices to the cluster
		"""
		indices = np.where((self.flag & (PointFlag.NOISE + PointFlag.GROUND)) == 0)[0]
		if len(indices) < 1:
			return None

		vox = np.unique(np.c_[self.ix[indices], self.iy[indices], self.ih[indices]], axis=0)

		# Get new_indices for all vox in one step
		new_indices = np.any((self.ix[:, None] == vox[:, 0]) & (self.iy[:, None] == vox[:, 1])\
		        & (self.ih[:, None] > vox[:, 2]-3) & (self.ih[:, None] < vox[:, 2]), axis=1)
		
		# Update flag
		self.flag[new_indices] &= (255 - PointFlag.GROUND)


	def height_gridding(self, t_pad):
		"""
		After the height calculation is done, gridding the height
		"""
		self.ih = (self.hs / t_pad.cfg["h_grid_size"]).astype(np.int16)


	def visualize(self, psize=2):
		is_noise = self.flag & PointFlag.NOISE > 0
		# Add the noise points
		data = [go.Scatter3d(x=self.pos[is_noise, 0],
												 y=self.pos[is_noise, 1],
												 z=self.pos[is_noise, 2],
												 mode = 'markers', name = f"{is_noise.sum()} Noise Points",
												 marker=dict(size=psize, opacity=0.3, color='purple'))]
		
		# Add the ground points
		is_grd = self.flag & PointFlag.GROUND > 0
		data.append(go.Scatter3d(x=self.pos[is_grd, 0],
				 										 y=self.pos[is_grd, 1],
														 z=self.pos[is_grd, 2],
														 mode = 'markers', name = f"{is_grd.sum()} Ground Points",
														 marker=dict(size=psize, opacity=0.3, color='green')))
		
		# BKG points
		is_bkg = self.flag & PointFlag.BKG > 0
		data.append(go.Scatter3d(x=self.pos[is_bkg, 0],
				 										 y=self.pos[is_bkg, 1],
														 z=self.pos[is_bkg, 2],
														 mode = 'markers', name = f"{is_bkg.sum()} Background Points",
														 marker=dict(size=psize, opacity=0.3, color='lightgray')))
		
		# Draw the targets and add annotations
		annotations = []
		sn_set = np.unique(self.sn_arr[self.sn_arr > 0])
		for sn in sn_set:
			pick = self.sn_arr == sn
			pos_avg = self.pos[pick].mean(axis=0)

			cls_type = "unknown"
			if self.flag[pick][0] & PointFlag.VEHCLE > 0:
				cls_type = "vehicle"
			elif self.flag[pick][0] & PointFlag.CYCLIST > 0:
				cls_type = "cyclist"
			elif self.flag[pick][0] & PointFlag.PED > 0:
				cls_type = "pedestrian"

			annotations.append(dict(x=pos_avg[0], y=pos_avg[1], z=pos_avg[2]+0.4,
				 text=f"{cls_type}{sn}", xanchor='left', showarrow=False,
				 font=dict(size=12, color='white')))

		# Add the other points
		others = ~(is_grd | is_noise | is_bkg)
		data.append(go.Scatter3d(x=self.pos[others, 0],
				 										 y=self.pos[others, 1],
														 z=self.pos[others, 2],
														 mode = 'markers', name = f"{others.sum()} Other Points",
														 marker=dict(size=psize, opacity=0.3, color='red')))

		fig = go.Figure(data=data)
		fig.update_layout(title="frame", template='plotly_dark',
				scene=dict(annotations=annotations, aspectmode='data'))
		fig.show()
