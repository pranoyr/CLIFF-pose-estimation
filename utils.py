camera_distance = 10
import torch
import numpy as np

from enum import Enum

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

 
import numpy as np

import torch
import torch.nn as nn




device = torch.device("cuda:1")


body_idx = np.squeeze(np.arange(24)) 
body_idx = torch.from_numpy(body_idx).to(device).long()

def perspective_projection(focal=1015., center=112.):
	# return p.T (N, 3) @ (3, 3) 
	return torch.tensor([
		focal, 0, center,
		0, focal, center,
		0, 0, 1
	]).reshape([3, 3]).type(torch.float32).t().to(device)



def to_image(face_shape, focal, center):
		"""
		Return:
			face_proj        -- torch.tensor, size (B, N, 2), y direction is opposite to v direction

		Parameters:
			face_shape       -- torch.tensor, size (B, N, 3)
		"""
		# to image_plane
		face_proj = face_shape @ perspective_projection(focal, center)
		face_proj = face_proj[..., :2] / face_proj[..., 2:]

		return face_proj

def to_camera(face_shape):
		face_shape[..., -1] = -camera_distance - face_shape[..., -1]
		return face_shape



def get_landmarks(face_proj):
		"""
		Return:
			face_lms         -- torch.tensor, size (B, 68, 2)

		Parameters:
			face_proj       -- torch.tensor, size (B, N, 2)
		"""  
		return face_proj[:, body_idx]
		# return face_proj


def perspective_projection(points, rotation, translation,
						   focal_length, camera_center):
	"""
	This function computes the perspective projection of a set of points.
	Input:
		points (bs, N, 3): 3D points
		rotation (bs, 3, 3): Camera rotation
		translation (bs, 3): Camera translation
		focal_length (bs,) or scalar: Focal length
		camera_center (bs, 2): Camera center
	"""
	batch_size = points.shape[0]
	K = torch.zeros([batch_size, 3, 3], device=points.device)
	K[:,0,0] = focal_length
	K[:,1,1] = focal_length
	K[:,2,2] = 1.
	K[:,:-1, -1] = camera_center



	# Transform points
	points = torch.einsum('bij,bkj->bki', rotation, points)
	#points = points + translation.unsqueeze(1)

	# Apply perspective distortion
	projected_points = points / points[:,:,-1].unsqueeze(-1)

	# Apply camera intrinsics
	projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

	return projected_points[:, :, :-1]




class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda:1" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def to_tensor(tensor, dtype=torch.float32):
    if torch.Tensor == type(tensor):
        return tensor.clone().detach()
    else:
        return torch.tensor(tensor, dtype)


def rel_change(prev_val, curr_val):
    return (prev_val - curr_val) / max([np.abs(prev_val), np.abs(curr_val), 1])


def max_grad_change(grad_arr):
    return grad_arr.abs().max()


class JointMapper(nn.Module):
    def __init__(self, joint_maps=None):
        super(JointMapper, self).__init__()
        if joint_maps is None:
            self.joint_maps = joint_maps
        else:
            self.register_buffer('joint_maps',
                                 torch.tensor(joint_maps, dtype=torch.long))

    def forward(self, joints, **kwargs):
        if self.joint_maps is None:
            return joints
        else:
            return torch.index_select(joints, 1, self.joint_maps)


class GMoF(nn.Module):
    def __init__(self, rho=1):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist



def get_joint_weights():
        # The weights for the joint terms in the optimization
        optim_weights = np.ones(self.num_joints + 2 * self.use_hands +
                                self.use_face * 51 +
                                17 * self.use_face_contour,
                                dtype=np.float32)

        # Neck, Left and right hip
        # These joints are ignored because SMPL has no neck joint and the
        # annotation of the hips is ambiguous.
        if self.joints_to_ign is not None and -1 not in self.joints_to_ign:
            optim_weights[self.joints_to_ign] = 0.
        return torch.tensor(optim_weights, dtype=self.dtype)



def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))
