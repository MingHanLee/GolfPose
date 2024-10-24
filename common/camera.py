import numpy as np
import torch

from common.utils import wrap
from common.quaternion import qrot, qinverse
import cv2

def normalize_screen_coordinates(X, w, h): 
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X/w*2 - [1, h/w]

    
def image_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Reverse camera frame normalization
    return (X + [1, h/w])*w/2
    

def world_to_camera(X, R, t):
    Rt = wrap(qinverse, R) # Invert rotation
    return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t) # Rotate and translate

    
def camera_to_world(X, R, t):
    return wrap(qrot, np.tile(R, (*X.shape[:-1], 1)), X) + t

    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2
    
    return f*XXX + c

def project_to_2d_linear(X, camera_params):
    """
    Project 3D points to 2D using only linear parameters (focal length and principal point).
    
    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]
    
    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)
        
    f = camera_params[..., :2]
    c = camera_params[..., 2:4]
    
    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    
    return f*XX + c

def uvd2xyz(uvd, gt_3D, cam):
    """
    transfer uvd to xyz
    :param uvd: N*T*V*3 (uv and z channel)
    :param gt_3D: N*T*V*3 (NOTE: V=0 is absolute depth value of root joint)
    :return: root-relative xyz results
    """
    N, T, V,_ = uvd.size()


    dec_out_all = uvd.view(-1, T, V, 3).clone()  # N*T*V*3
    root = gt_3D[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1).clone()# N*T*V*3
    enc_in_all = uvd[:, :, :, :2].view(-1, T, V, 2).clone()  # N*T*V*2

    cam_f_all = cam[..., :2].view(-1,1,1,2).repeat(1,T,V,1) # N*T*V*2
    cam_c_all = cam[..., 2:4].view(-1,1,1,2).repeat(1,T,V,1)# N*T*V*2

    # change to global
    z_global = dec_out_all[:, :, :, 2]# N*T*V
    z_global[:, :, 0] = root[:, :, 0, 2]
    z_global[:, :, 1:] = dec_out_all[:, :, 1:, 2] + root[:, :, 1:, 2]  # N*T*V
    z_global = z_global.unsqueeze(-1)  # N*T*V*1
    
    uv = enc_in_all - cam_c_all  # N*T*V*2
    xy = uv * z_global.repeat(1, 1, 1, 2) / cam_f_all  # N*T*V*2
    xyz_global = torch.cat((xy, z_global), -1)  # N*T*V*3
    xyz_offset = (xyz_global - xyz_global[:, :, 0, :].unsqueeze(-2).repeat(1, 1, V, 1))# N*T*V*3


    return xyz_offset


## added by yuchen
def project_to_2d_golf(X, mtx, dist):
    result = []
    f_num, j_num, d_num = X.shape

    for i in range(f_num):
        j_result = []
        for j in range(j_num):
            imgPoint, _ = cv2.projectPoints(X[i][j], np.array([0., 0., 0.]), np.array([0., 0., 0.]), mtx, dist)
            j_result.append(imgPoint.reshape(2))
        
        result.append(j_result)
    
    return np.array(result).astype('float32')

# rvec and tvec are calibrated from cv2.calibration
def world_to_camera_golf(X, rvec, tvec): # tvec is millimeter
    rvec, tvec = np.array(rvec), np.array(tvec).reshape(3, 1)
    R = cv2.Rodrigues(rvec)
    extrinsic = np.hstack((R[0], tvec))

    X = np.pad(X, ((0,0), (0,0), (0,1)), mode='constant', constant_values=1)
    
    result = []
    f_num, j_num, d_num = X.shape

    for i in range(f_num):
        j_result = []
        for j in range(j_num):
            j_result.append(np.dot(extrinsic, X[i][j]))
        result.append(j_result)

    return np.array(result).astype('float32')

def camera_to_world_golf(X, rvec, tvec): # tvec is millimeter
    rvec, tvec = np.array(rvec), np.array(tvec)
    R = cv2.Rodrigues(rvec)
    
    result = []
    f_num, j_num, d_num = X.shape

    for i in range(f_num):
        j_result = []
        for j in range(j_num):
            j_result.append(np.dot(R[0].T, X[i][j] - tvec))
        result.append(j_result)

    return np.array(result).astype('float32')

def vicon_to_world_golf(dots, base_dots, square_size):
    b, c, a = np.asarray(base_dots[0]), np.asarray(base_dots[1]), np.asarray(base_dots[2])
    x_vec = c - b
    y_vec = a - b
    z_vec = np.cross(x_vec, y_vec)

    x_norm = np.linalg.norm(x_vec)
    y_norm = np.linalg.norm(y_vec)
    z_norm = np.linalg.norm(z_vec)

    result = []
    for frame_dots in dots:
        frame_result = []
        for dot in frame_dots:
            dot = np.asarray(dot)
            dot_vec = dot - b
            # dot_vec.shape = (3,) but shape = (22,3)
            proj_x = (np.dot(dot_vec, x_vec) / x_norm**2) * x_vec
            proj_y = (np.dot(dot_vec, y_vec) / y_norm**2) * y_vec
            proj_z = (np.dot(dot_vec, z_vec) / z_norm**2) * z_vec

            x_coord = np.linalg.norm(proj_x)
            y_coord = np.linalg.norm(proj_y)
            z_coord = np.linalg.norm(proj_z)

            if not all(np.sign(x_vec)==np.sign(proj_x)): x_coord *= -1
            if not all(np.sign(y_vec)==np.sign(proj_y)): y_coord *= -1
            if not all(np.sign(z_vec)==np.sign(proj_z)): z_coord *= -1

            x_coord -= square_size

            frame_result.append([x_coord, y_coord, z_coord])

        result.append(frame_result)

    return np.array(result).astype('float32')


def world_to_vicon_golf(dots, base_dots):
    o, x, y, z = np.asarray(base_dots[0]), np.asarray(base_dots[1]), np.asarray(base_dots[2]), np.asarray(base_dots[3])
    x_vec = x - o
    y_vec = y - o
    z_vec = z - o

    x_norm = np.linalg.norm(x_vec)
    y_norm = np.linalg.norm(y_vec)
    z_norm = np.linalg.norm(z_vec)

    result = []
    for frame_dots in dots:
        frame_result = []
        for dot in frame_dots:
            dot = np.asarray(dot)
            dot_vec = dot - o

            proj_x = (np.dot(dot_vec, x_vec) / x_norm**2) * x_vec
            proj_y = (np.dot(dot_vec, y_vec) / y_norm**2) * y_vec
            proj_z = (np.dot(dot_vec, z_vec) / z_norm**2) * z_vec

            x_coord = np.linalg.norm(proj_x)
            y_coord = np.linalg.norm(proj_y)
            z_coord = np.linalg.norm(proj_z)

            if not all(np.sign(x_vec)==np.sign(proj_x)): x_coord *= -1
            if not all(np.sign(y_vec)==np.sign(proj_y)): y_coord *= -1
            if not all(np.sign(z_vec)==np.sign(proj_z)): z_coord *= -1

            frame_result.append([x_coord, y_coord, z_coord])

        result.append(frame_result)

    return np.array(result).astype('float32')