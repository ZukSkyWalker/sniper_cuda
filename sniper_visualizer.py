import numpy as np
import pandas as pd
import os
import open3d as o3d
from matplotlib import cm
# from scipy.spatial import ConvexHull, QhullError

from src.frame import PointFlag


def get_fiducial(rc, sn):
	"""
	Read the npz file, get (z0, z1) and vertices for the given sn 
	"""

	# Get the raw points
	sel = rc['sn'] == sn
	if sel.sum() < 1:
		return None, None
	
	z0, z1 = rc['pos'][sel, 2].min(), rc['pos'][sel, 2].max()
	
	# Get convexhull vertices
	vertices_xy = rc['fid_vertices'][rc['fiducial_sn']==sn]
	if len(vertices_xy) < 3:
		return None, None
	
	num_vertices = vertices_xy.shape[0]

	vertices_lo = np.c_[vertices_xy, z0 + np.zeros(num_vertices)]
	vertices_hi = np.c_[vertices_xy, z1 + np.zeros(num_vertices)]

	# Prepare the lineset
	line_set = o3d.geometry.LineSet()
	vertices = np.vstack([vertices_lo, vertices_hi])
	line_set.points = o3d.utility.Vector3dVector(vertices)

	lines_lower = [[i, (i + 1) % num_vertices] for i in range(num_vertices)]
	lines_upper = [[i + num_vertices, (i + 1) % num_vertices + num_vertices] for i in range(num_vertices)]
	lines_vertical = [[i, i + num_vertices] for i in range(num_vertices)]
	lines = lines_lower + lines_upper + lines_vertical
	line_set.lines = o3d.utility.Vector2iVector(lines)
	
	return line_set


def tracking_info(df, f_idx):
	cols = ['sn', 'x', 'y', 'npts', 'category']
	os.system('clear')
	print(f"Frame {f_idx}")
	df_sample = df[(df.frame == f_idx)][cols].to_string(index=False)
	print(df_sample)

class vis_pad():
	def __init__(self, frm) -> None:
		self.vis = o3d.visualization.Visualizer()
		self.vis.create_window()
		render_options = self.vis.get_render_option()
		render_options.point_size = 2.0
		
		# Set the background to black
		opt = self.vis.get_render_option()
		opt.background_color = np.asarray([0, 0, 0])

		# Initialize the ground point cloud
		is_grd = frm['f'] & PointFlag.GROUND > 0
		self.grd_pcd = o3d.geometry.PointCloud()
		self.grd_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_grd])
		ref = frm['r'][is_grd].clip(0, 1)
		self.grd_pcd.colors = o3d.utility.Vector3dVector(cm.jet(ref)[:,:3])
		self.vis.add_geometry(self.grd_pcd)

		# Initialize the bkg cloud
		is_bkg = frm['f'] & PointFlag.BKG > 0
		self.bkg_pcd = o3d.geometry.PointCloud()
		self.bkg_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_bkg])
		## RGB: white
		bkg_col = 0.5+np.zeros((is_bkg.sum(), 3))
		self.bkg_pcd.colors = o3d.utility.Vector3dVector(bkg_col)
		self.vis.add_geometry(self.bkg_pcd)

		# Initialize the vehicle points
		is_veh = frm['f'] & PointFlag.VEHCLE > 0
		self.veh_pcd = o3d.geometry.PointCloud()
		self.veh_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_veh])
		veh_col = np.zeros((is_veh.sum(), 3))
		veh_col[:, 0] = 1 # Mark it red (1, 0, 0)
		self.veh_pcd.colors = o3d.utility.Vector3dVector(veh_col)
		self.vis.add_geometry(self.veh_pcd)

		# Initialize the pedestrian points
		is_ped = frm['f'] & PointFlag.PED > 0
		self.ped_pcd = o3d.geometry.PointCloud()
		self.ped_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_ped])
		ped_col = np.zeros((is_ped.sum(), 3))
		ped_col[:, :2] = 1 # Mark pedestrian yellow (1, 1, 0)
		self.ped_pcd.colors = o3d.utility.Vector3dVector(ped_col)
		self.vis.add_geometry(self.ped_pcd)

		# Initialize the cyclist points
		is_cyc = frm['f'] & PointFlag.CYCLIST > 0
		self.cyc_pcd = o3d.geometry.PointCloud()
		self.cyc_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_cyc])
		cyc_col = np.zeros((is_cyc.sum(), 3))
		cyc_col[:, 1] = 1 # Mark cyclist green (0, 1, 0)
		self.cyc_pcd.colors = o3d.utility.Vector3dVector(cyc_col)
		self.vis.add_geometry(self.cyc_pcd)

		# Initialize the other signal points
		is_sig = (frm['sn'] > 0) & (~is_veh) & (~is_ped) & (~is_cyc)
		self.sig_pcd = o3d.geometry.PointCloud()
		self.sig_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_sig])
		sig_col = np.zeros((is_sig.sum(), 3))
		sig_col[:, 1:] = 1  # Mark the other points cyan
		self.sig_pcd.colors = o3d.utility.Vector3dVector(sig_col)
		self.vis.add_geometry(self.sig_pcd)

		# Initialize the bounding boxes
		# self.bboxes = []

	def update_pcd(self, frm):
		# Initialize the vehicle points
		is_veh = frm['f'] & PointFlag.VEHCLE > 0
		self.veh_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_veh])
		veh_col = np.zeros((is_veh.sum(), 3))
		veh_col[:, 0] = 1 # Mark it red (1, 0, 0)
		self.veh_pcd.colors = o3d.utility.Vector3dVector(veh_col)
		self.vis.update_geometry(self.veh_pcd)

		# Initialize the pedestrian points
		is_ped = frm['f'] & PointFlag.PED > 0
		self.ped_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_ped])
		ped_col = np.zeros((is_ped.sum(), 3))
		ped_col[:, :2] = 1 # Mark pedestrian yellow (1, 1, 0)
		self.ped_pcd.colors = o3d.utility.Vector3dVector(ped_col)
		self.vis.update_geometry(self.ped_pcd)

		# Initialize the cyclist points
		is_cyc = frm['f'] & PointFlag.CYCLIST > 0
		# self.cyc_pcd = o3d.geometry.PointCloud()
		self.cyc_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_cyc])
		cyc_col = np.zeros((is_cyc.sum(), 3))
		cyc_col[:, 1] = 1 # Mark cyclist green (0, 1, 0)
		self.cyc_pcd.colors = o3d.utility.Vector3dVector(cyc_col)
		self.vis.update_geometry(self.cyc_pcd)

		# Initialize the other signal points
		is_sig = (frm['sn'] > 0) & (~is_veh) & (~is_ped) & (~is_cyc)
		# self.sig_pcd = o3d.geometry.PointCloud()
		self.sig_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_sig])
		sig_col = np.zeros((is_sig.sum(), 3))
		sig_col[:, 1:] = 1 # Mark the other signal points cyan (0, 1, 1)
		self.sig_pcd.colors = o3d.utility.Vector3dVector(sig_col)
		self.vis.update_geometry(self.sig_pcd)

		print(f"{is_veh.sum()} veh points; {is_cyc.sum()} cyc points; {is_ped.sum()} ped points; {is_sig.sum()} other points")


	def update_labels(self, df_labels):
		# remove the old labels
		for text in self.labels:
			self.vis.remove_geometry(text)

		self.labels = []

		# Add new labels
		for _, row in df_labels.iterrows():
			prob, category = row['prob_veh'], "vehicle"

			if row["prob_ped"] > prob:
				prob = row["prob_ped"]
				category = "pedestrian"

			if row["prob_cyc"] > prob:
				prob = row["prob_cyc"]
				category = "cyclist"

			label = f"{row['category']} {row['sn']}\n: {100*prob:.1f}% {category}"
			text_3d = o3d.geometry.Text3D(label, [row['x'], row['y'], row['z']], size=1, orientation=[0,0,0])
			self.vis.add_geometry(text_3d)


	def update_bboxes(self, frm):
		for box in self.bboxes:
			self.vis.remove_geometry(box)

		self.bboxes = []

		for sn in np.unique(frm['sn']):
			if sn < 1:
				continue

			line_set = get_fiducial(frm, sn)
			self.vis.add_geometry(line_set)
			self.bboxes.append(line_set)


	def update_frame(self, frm):
		# Update the ground points
		is_grd = frm['f'] & PointFlag.GROUND > 0
		self.grd_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_grd])
		ref = frm['r'][is_grd].clip(0, 1)
		self.grd_pcd.colors = o3d.utility.Vector3dVector(cm.jet(ref)[:,:3])
		self.vis.update_geometry(self.grd_pcd)

		# Update the bkg points
		is_bkg = frm['f'] & PointFlag.BKG > 0
		self.bkg_pcd.points = o3d.utility.Vector3dVector(frm['pos'][is_bkg])
		## RGB: white
		bkg_col = 0.5+np.zeros((is_bkg.sum(), 3))
		self.bkg_pcd.colors = o3d.utility.Vector3dVector(bkg_col)
		self.vis.update_geometry(self.bkg_pcd)

		# Update the signal points
		self.update_pcd(frm)

		# Refresh bounding boxes
		# self.update_bboxes(frm)
		
		self.vis.poll_events()
		self.vis.update_renderer()


if __name__ == "__main__":
	# Get the data source
	path = "../Data/sense33/"
	file_name = f"{path}reco/rc_0.npz"
	frm = np.load(file_name)

	df = pd.read_csv(f"{path}tracking.csv", index_col=False)
	df = df[df.npts > 0]

	# Initialize the vis pad
	vpad = vis_pad(frm)

	for i in range(6000):
		file_name = f"{path}reco/rc_{i}.npz"
		if not os.path.isfile(file_name):
			break
		frm = np.load(file_name)
		vpad.update_frame(frm)

		tracking_info(df, i)
	
	vpad.vis.destroy_window()
