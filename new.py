import torch
import torch.nn as nn
import torch.nn.functional as F
def mat_3x3_inv(mat):
    '''
    calculate the inverse of a 3x3 matrix, support batch.
    :param mat: torch.Tensor -- [input matrix, shape: (B, 3, 3)]
    :return: mat_inv: torch.Tensor -- [inversed matrix shape: (B, 3, 3)]
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    # Divide the matrix with it's maximum element
    max_vals = mat.max(1)[0].max(1)[0].view((-1, 1, 1))
    mat = mat / max_vals

    det = mat_3x3_det(mat)
    inv_det = 1.0 / det

    mat_inv = torch.zeros(mat.shape, device=mat.device)
    mat_inv[:, 0, 0] = (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 0, 1] = (mat[:, 0, 2] * mat[:, 2, 1] - mat[:, 0, 1] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 0, 2] = (mat[:, 0, 1] * mat[:, 1, 2] - mat[:, 0, 2] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 1, 0] = (mat[:, 1, 2] * mat[:, 2, 0] - mat[:, 1, 0] * mat[:, 2, 2]) * inv_det
    mat_inv[:, 1, 1] = (mat[:, 0, 0] * mat[:, 2, 2] - mat[:, 0, 2] * mat[:, 2, 0]) * inv_det
    mat_inv[:, 1, 2] = (mat[:, 1, 0] * mat[:, 0, 2] - mat[:, 0, 0] * mat[:, 1, 2]) * inv_det
    mat_inv[:, 2, 0] = (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 2, 0] * mat[:, 1, 1]) * inv_det
    mat_inv[:, 2, 1] = (mat[:, 2, 0] * mat[:, 0, 1] - mat[:, 0, 0] * mat[:, 2, 1]) * inv_det
    mat_inv[:, 2, 2] = (mat[:, 0, 0] * mat[:, 1, 1] - mat[:, 1, 0] * mat[:, 0, 1]) * inv_det

    # Divide the maximum value once more
    mat_inv = mat_inv / max_vals
    return mat_inv
def mat_3x3_det(mat):
    '''
    calculate the determinant of a 3x3 matrix, support batch.
    '''
    if len(mat.shape) < 3:
        mat = mat[None]
    assert mat.shape[1:] == (3, 3)

    det = mat[:, 0, 0] * (mat[:, 1, 1] * mat[:, 2, 2] - mat[:, 2, 1] * mat[:, 1, 2]) \
        - mat[:, 0, 1] * (mat[:, 1, 0] * mat[:, 2, 2] - mat[:, 1, 2] * mat[:, 2, 0]) \
        + mat[:, 0, 2] * (mat[:, 1, 0] * mat[:, 2, 1] - mat[:, 1, 1] * mat[:, 2, 0])
    return det
def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)
def compute_D(points, norm):
    """
    inputs:
        points            b, 4, H*W
        norm              b, 3, H, W
    outputs:
        D                      b, 1, H, W
    """
    batch_size = points.shape[0]
    norm = norm.reshape(batch_size, 3, -1).permute(0, 2, 1).unsqueeze(2) # b , H*W, 1, 3
    points = points[:, :3, :].permute(0, 2, 1).unsqueeze(3) # b, H*W, 3, 1
    points = points.float()
    D = - norm @ points  # b, H*W

    return D


def depth2norm(cam_points, height, width, nei=3):
    pts_3d_map = cam_points[:, :3, :].permute(0, 2, 1).view(-1, height, width, 3)

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:, nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:, nei:-nei, 0:-(2 * nei), :]
    pts_3d_map_y0 = pts_3d_map[:, 0:-(2 * nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:, nei:-nei, 2 * nei:, :]
    pts_3d_map_y1 = pts_3d_map[:, 2 * nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:, 0:-(2 * nei), 0:-(2 * nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:, 2 * nei:, 0:-(2 * nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:, 0:-(2 * nei), 2 * nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:, 2 * nei:, 2 * nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    diff_x0 = diff_x0.reshape(-1, 3)
    diff_y0 = diff_y0.reshape(-1, 3)
    diff_x1 = diff_x1.reshape(-1, 3)
    diff_y1 = diff_y1.reshape(-1, 3)
    diff_x0y0 = diff_x0y0.reshape(-1, 3)
    diff_x0y1 = diff_x0y1.reshape(-1, 3)
    diff_x1y0 = diff_x1y0.reshape(-1, 3)
    diff_x1y1 = diff_x1y1.reshape(-1, 3)

    ## calculate normal by cross product of two vectors
    normals0 = torch.cross(diff_x1, diff_y1)
    normals1 = torch.cross(diff_x0, diff_y0)
    normals2 = torch.cross(diff_x0y1, diff_x0y0)
    normals3 = torch.cross(diff_x1y0, diff_x1y1)

    normal_vector = normals0 + normals1 + normals2 + normals3
    normal_vectorl2 = torch.norm(normal_vector, p=2, dim=1)
    normal_vector = torch.div(normal_vector.permute(1, 0), normal_vectorl2)
    normal_vector = normal_vector.permute(1, 0).view(pts_3d_map_ctr.shape).permute(0, 3, 1, 2)
    normal_map = F.pad(normal_vector, (0, 2 * nei, 2 * nei, 0), "constant", value=0)
    normal = - F.normalize(normal_map, dim=1, p=2)
    return normal


def compute_mmap(batch_size, norm, vps, H, W, epoch, nei):
    """
    inputs:
        norm                           b,3,H,W       tensor
        vps                               b,6,3             tensor
    outputs:
        mmap                         b,1,H,W       tensor
        mmap_mask           b,1,H,W       tensor
    """
    norm_flatten = norm.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)  # bxNx3
    vps_6 = vps.repeat((norm_flatten.shape[1], 1, 1, 1)).permute(1, 2, 0, 3)  # bx6xNx3
    norm_flatten_6 = norm_flatten.repeat((6, 1, 1, 1)).permute(1, 0, 2, 3)  # bx6xNx3
    cos = nn.CosineSimilarity(dim=3, eps=1e-6)
    cos_sim = cos(vps_6, norm_flatten_6)
    score, index = torch.max(cos_sim, 1)
    mmap = index.reshape(batch_size, 1, H, W)
    score_map = score.reshape(batch_size, 1, H, W)
    '''
    When the estimated normal is very close to the given principal direction, \
    NaN with cos greater than 1 will appear, so NaN will be set to 1 here.
    '''
    if torch.any(torch.isnan(score_map)):
        print('nan in mmap compute! set nan = 1')
        torch.nan_to_num(score_map, nan=1)

    # The mask here first comes from the top edge and the right edge in depth2norm.
    mmap_mask = torch.ones_like(mmap).cuda()
    mmap_mask[:, :, :20, :] = 0
    mmap_mask[:, :, -8:, :] = 0
    mmap_mask[:, :, :, :8] = 0
    mmap_mask[:, :, :, -8:] = 0
    '''
    Secondly, an adaptive threshold is used to filter the pixels with too large an Angle deviation,\
    with an initial Angle of about 25 degrees
    '''
    score = 1.633 * epoch + 900
    mmap_mask[1000 * score_map < score] = 0

    return mmap, mmap_mask, score

def align_smooth_norm(batch_size, mmap, vps, H, W):
    """
    inputs:
        mmap                            b, 1, H, W           tensor
        vps                                  b, 6, 3                 tensor
    outputs:
        smooth_norm            b, 3, H*W            tensor
    """
    mmap_label = mmap.reshape(batch_size, -1)
    mmap_label = mmap_label.repeat(3,1,1).permute(1,0,2)
    vps = vps.permute(0, 2, 1)
    smooth_norm = torch.gather(vps, 2, mmap_label)
    smooth_norm = smooth_norm.reshape(batch_size, 3, H, W)

    return smooth_norm