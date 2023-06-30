import torch
import numpy as np
import src.util as util

def get_prob_vec(pos, t_pad):
  """
  Input:
    pos is a numpy array
  Output:
    probability array: [vehicle, pedestrian, cyclist]
  """
  # Dimension check
  height = pos[:, 2].max() - pos[:, 2].min()

  if (height > t_pad.cfg["max_sig_height"]) | (height < t_pad.cfg["local_dz_max"]):
    return np.zeros(3)
  radius = util.get_radius(pos[:, :2])
  if (radius > t_pad.cfg["max_sig_radius"]) | (radius < t_pad.cfg["min_sig_radius"]):
    return np.zeros(3)
  
  # print(f"radius = {radius:.2f}, height={height:.2f}")

  pos_tensor = torch.from_numpy(pos.copy())
  # shift to the center:
  # pos_tensor -= 0.5 * (pos_tensor.min(dim=0).values + pos_tensor.max(dim=0).values)

  pos_tensor -= pos_tensor.mean(dim=0)

  with torch.no_grad():
    logits, _, _ = t_pad.model(pos_tensor.unsqueeze(0))

  prob_vec = torch.sigmoid(logits)[0].numpy()

  # Rule based cut
  if (radius > t_pad.cfg["max_cyc_radius"]) | (height > t_pad.cfg["max_cyc_height"]):
    prob_vec[1:] = 0
  elif (radius > t_pad.cfg["max_ped_radius"]) | (height > t_pad.cfg["max_ped_height"]):
    prob_vec[1] = 0

  norm = prob_vec.sum()

  if norm > 1:
    prob_vec /= norm

  return prob_vec
