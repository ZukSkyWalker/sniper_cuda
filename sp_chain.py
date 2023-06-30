"""@Reconstruction Script
Autonomous Perception

=============================================
Author: zukai.wang@cepton.com
Date: 09/18/2022
Version: 1.0.0
=============================================
"""

import numpy as np
from optparse import OptionParser


import numpy as np
import json
import os
import torch
import logging

from src.frame import Frame
from src.track_pad import Pad
import cepton_sdk2.utils as utils
from sniper_classifier import SniperNet
import src.util as util


def trackpad_init(folder_path):
	# 1. Initialize track pad
	with open("cfg/cfg.json") as f:
		cfg = json.load(f)

	model = SniperNet(3)
	model = torch.load("cfg/model.pt", map_location=torch.device("cpu"))
	model.eval()

	t_pad = Pad(cfg, model)

	# Load ground surface matrix
	grd_path = f"{folder_path}gs.npz"
	if os.path.exists(grd_path):
		t_pad.load_ground_surf(surf_pts=np.load(grd_path))

	# Load background matrix
	bkg_path = f"{folder_path}bkg_prob.npz"
	if os.path.exists(bkg_path):
		t_pad.load_bkg(bkg_npz=np.load(bkg_path))

	return t_pad


if __name__ == "__main__":
	# 0. Options Parsing
	parser = OptionParser()

	parser.add_option("-d", "--dir", dest="path", default="../Data/sense33/",
										help="input file path", type="str")
	
	parser.add_option("-n", "--num", dest="n_frames", default=100,
										help="max number of frames", type="int")

	(options, args) = parser.parse_args()

	# 1. Initialize track pad
	t_pad = trackpad_init(options.path)

	# 2. Loading frames and process
	f_idx = 0
	# Start track Logging Format
	logging.basicConfig(
			filename=f'{options.path}tracking.csv', level=logging.DEBUG, filemode='w+')
	cols = (
		",frame,t,sn,npts,x,y,z,z0,z1,vx,vy,"
		+ "prob_veh,prob_ped,prob_cyc,"
    + "category,is_dynamic,is_static"
	)
	logging.info(cols)

	for rf in utils.ReadPcap(f"{options.path}raw.pcap"):
		print(f"Processing frame {f_idx}")
		if t_pad.t0 < 1:
			t_pad.t0 = rf.timestamps[0]
		
		frm = Frame(rf, t_pad)
		t_pad.frame_processing(frm)

		# Container for the fiducial volumes
		fiducial_vertices = []
		fiducial_sn = []

		# Log the track list
		frame_info = f",{f_idx},{t_pad.timestamp},"
		for sn in t_pad.track_dict:
			z0, z1 = 0, 0
			sel = frm.sn_arr==sn
			if sel.sum() > 0:
				z0 = frm.pos[sel, 2].min()
				z1 = frm.pos[sel, 2].max()

			tgt = t_pad.track_dict[sn]
			track_info  = f"{sn},{tgt.n_pts},{tgt.pos_last[0]:.2f},{tgt.pos_last[1]:.2f},{tgt.pos_last[2]:.2f},"
			track_info += f"{z0:.2f},{z1:.2f},"
			track_info += f"{tgt.velocity[0]:.2f},{tgt.velocity[1]:.2f},"
			track_info += f"{tgt.prob_vec[0]:.2f},{tgt.prob_vec[1]:.2f},{tgt.prob_vec[2]:.2f},"
			track_info += f"{tgt.category},{tgt.is_dynamic},{tgt.is_static}"

			logging.info(frame_info+track_info)

			# Prepare the fiducial vertices
			xy_pos = frm.pos[sel, :2]
			if len(xy_pos) < 1:
				continue

			# append more sn indices
			ix_arr, iy_arr = np.where(t_pad.sn_map == sn)
			if len(ix_arr) > 0:
				rho = (iy_arr + 0.5) * t_pad.cfg["dist_grid"]
				phi = (ix_arr + 0.5) * t_pad.cfg["d_phi"] - t_pad.cfg["phi_max"]
				xy_pos = np.vstack([xy_pos, np.c_[rho * np.sin(phi), rho * np.cos(phi)]])

			# Get the vertices of the object
			vertices = util.get_convex_vertices(xy_pos)
			if vertices is None:
				continue
			fiducial_vertices.append(vertices)
			sn_arr = np.zeros(vertices.shape[0]) + sn
			fiducial_sn.append(sn_arr.astype(np.uint32))

		# uniq_sns = np.unique(frm.sn_arr)
		# uniq_sns = uniq_sns[uniq_sns > 0]		
		# print(f"{len(uniq_sns)} tracked objects, sn range: {uniq_sns.min()}, {uniq_sns.max()}")

		map_ix, map_iy = np.where(t_pad.sn_map > 0)

		fiducial_vertices_arr = np.array([])
		fiducial_sn_arr = np.array([])

		if len(fiducial_vertices) > 0:
			fiducial_vertices_arr = np.vstack(fiducial_vertices)
			fiducial_sn_arr = np.hstack(fiducial_sn)

		np.savez(
				f'{options.path}reco/rc_{f_idx}.npz',
					t=frm.ts,
					f=frm.flag,
					r=frm.rs,
					pos=frm.pos,
					h=frm.hs,
					cid=frm.cid_arr,
					sn=frm.sn_arr,
					pr=frm.bkg_probs,
					map_ix=map_ix,
					map_iy=map_iy,
					map_sn=t_pad.sn_map[map_ix, map_iy],
					fid_vertices=fiducial_vertices_arr,
					fiducial_sn=fiducial_sn_arr)
		f_idx += 1

		if f_idx > options.n_frames:
			break

		print()

	# Save the ground surface matrix
	# t_pad.save_ground_surf(out_path=f"{options.path}gs.npz")
	# print("Ground surface matrix saved")
	# Save the bkg matrix
	# t_pad.save_bkg(out_path=f"{options.path}bkg_prob.npz")
	# print("Background matrix saved")
