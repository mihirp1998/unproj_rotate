import voxel
import pickle
import numpy as np
import ipdb
import tensorflow as tf
tf.enable_eager_execution()

st = ipdb.set_trace
# sess = tf.Session()
# st()
from nbtschematic import SchematicFile
import numpy as np
from scipy.misc import imsave
import binvox_rw
import pickle
fov = [40.0]
import binvox_rw


def save_voxel(voxel_, filename, THRESHOLD=0.5):
	S1 = voxel_.shape[2]
	S2 = voxel_.shape[1]
	S3 = voxel_.shape[0]
	# st()
	binvox_obj = binvox_rw.Voxels(
		np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
		dims = [S1, S2, S3],
		translate = [0.0, 0.0, 0.0],
		scale = 1.0,
		axis_order = 'xyz'
	)   

	with open(filename, "wb") as f:
		binvox_obj.write(f)




fovs_working ={}

def rotate_voxels(rep,angle,fov):
	a = binvox_rw.read_as_3d_array(open("unprojected_voxels/outline_scale_47.binvox","rb"))
	val = a.data
	val = tf.convert_to_tensor(np.expand_dims(np.expand_dims(val,0),-1))
	voxel.fov =fov
	phi,theta = angle
	proj_val = voxel.rotate_voxel(val,phi,theta)
	num = np.where(proj_val>0.5)[0]

	if len(num) > 0:
		print("found")
		fovs_working[fov] = len(num)
	proj_val = np.squeeze(proj_val)
	proj_val = proj_val >0.5
	proj_imgZ = np.mean(proj_val,0)
	
	imsave('{}/valRotate_phi_{}_theta_{}_fov_{:04d}_Z.png'.format(rep,phi,theta,fov), proj_imgZ)

	save_voxel(np.squeeze(proj_val),"{}/valRotate_THETA_{}_PHI_{}_fov_{}_.binvox".format(rep,theta[0],phi[0],fov))


rotate_voxels("rotated_voxels",([-20.0],[0.0]),47)