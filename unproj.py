from scipy.misc import imread
import ipdb
st = ipdb.set_trace
from scipy.misc import imsave
# from nets import unproject
import voxel
import numpy as np
# st()
import imageio
import tensorflow as tf
tf.enable_eager_execution()
import binvox_rw

H = 128


def save_voxel(voxel_, filename, THRESHOLD=0.5):
  S1 = voxel_.shape[2]
  S2 = voxel_.shape[1]
  S3 = voxel_.shape[0]

  binvox_obj = binvox_rw.Voxels(
    np.transpose(voxel_, [2, 1, 0]) >= THRESHOLD,
    dims = [S1, S2, S3],
    translate = [0.0, 0.0, 0.0],
    scale = 1.0,
    axis_order = 'xyz'
  )   

  with open(filename, "wb") as f:
    binvox_obj.write(f)



def unproject(inputs, resize = False):

    if resize:
        inputs = tf.image.resize(inputs, (const.S, const.S))
    size = int(inputs.shape[1])

    inputs = voxel.unproject_image(inputs)

    meshgridz = tf.range(size, dtype = tf.float32)
    meshgridz = tf.reshape(meshgridz, (1, size, 1, 1))
    meshgridz = tf.tile(meshgridz, (1, 1, size, size))
    meshgridz = tf.expand_dims(meshgridz, axis = 4) 
    meshgridz = (meshgridz + 0.5) / (size/2) - 1.0

    unprojected_depth = (tf.expand_dims(inputs[:,:,:,:,0], 4) - voxel.RADIUS) * (1/voxel.SCENE_SIZE)
    if H > 32:
        outline_thickness = 0.1
    else:
        outline_thickness = 0.2

    outline = tf.cast(tf.logical_and(
        unprojected_depth <= meshgridz,
        unprojected_depth + outline_thickness > meshgridz
    ), tf.float32)

    if True:
        return outline,unprojected_depth

    inputs_ = [inputs]
    if const.USE_MESHGRID:
        inputs_.append(meshgridz)
    if const.USE_OUTLINE:
        inputs_.append(outline)
    inputs = tf.concat(inputs_, axis = 4)
    return inputs

def run(ricson,fov):
    voxel.RADIUS = 13.0
    voxel.SCENE_SIZE =8.0
    voxel.NEAR = voxel.RADIUS - voxel.SCENE_SIZE
    voxel.FAR = voxel.RADIUS + voxel.SCENE_SIZE
    voxel.fov =fov
    voxel.W = 64.0
    depth = np.array(imageio.imread("CLEVR_64_36_MORE_OBJ_FINAL_SMALL/depth/train/CLEVR_new_000003/CLEVR_new_000003_0_20.exr", format='EXR-FI'))[:,:,0]
    depth = depth * (100 - 0) + 0
    depth.astype(np.float32)
    val = np.expand_dims(np.expand_dims(depth,axis=-1),0)
    val,unprojected_depth = unproject(val)
    unprojected_depth = np.squeeze(unprojected_depth)
    val = np.squeeze(val)
    save_voxel(val, "unprojected_voxels/outline_scale_{}.binvox".format(fov))
    save_voxel(unprojected_depth, "unprojected_voxels/unproj_depths_{}.binvox".format(fov))

run(False,47)
