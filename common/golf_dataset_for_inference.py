# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import copy
from common.skeleton import Skeleton
from common.mocap_dataset import MocapDataset
from common.camera import normalize_screen_coordinates, image_coordinates, vicon_to_world_golf
       
golf_skeleton = Skeleton(
    # parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15],  # 17+0
    # parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 18],  # 17+1
    # parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 18, 17],  # 17+2 
    # parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 18, 17, 18], # 17+3
    # parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 16, 17, 18, 19], # 17+4
    parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 18, 17, 18, 19, 19],  # 17+5
    joints_left = [4, 5, 6, 11, 12, 13],
    joints_right = [1, 2, 3, 14, 15, 16])


# golf_skeleton = Skeleton(  # for only club
#     parents = [-1, 0, 1, 2, 2],  # 5 kps
#     joints_left = [],
#     joints_right = [])


square_size = 90

vicon_to_world_basis_dots = {
    'G1': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ],
    'G2': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ],
    'G3': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ],
    'G4': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ],
    'G5': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ],
    'G6': [
        [
            (311.692108, -379.845001, 1465.381104), # origin
            (182.505264,-291.05304,312.387482),   # x+ direction
            (100.367401,213.056839,1535.629883) # y+ direction
        ],
        [
            (936.364258, -370.360809, 1506.602051), # origin
            (897.232605, -460.533051, 346.276031),   # x+ direction
            (854.614075, 255.576675, 1461.180298) # y+ direction
        ]
    ]
}

world_to_vicon_basis_dots = {
    'G1': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ],
    'G2': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ],
    'G3': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ],
    'G4': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ],
    'G5': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ],
    'G6': [
        [
            (1425.6128267516065, 297.05432968107186, -46.5116718865498),  # o
            (1392.3058344879134, 196.95496237718928, 234.3107415302986), # x+
            (1448.5051977279045, 577.8974923184999, 56.36960116137839),  # y+
            (1128.3476528454964, 330.329465521058, -70.05345981300886) # z+
        ],
        [
            (1414.0116367075414, 595.3734331515628, -816.065482214389),
            (1403.9303354146825, 556.6222565439148, -518.7505965185186),
            (1390.7809916884157, 892.0799737085533, -778.1761312308406),
            (1115.0823733420455, 573.8426388739283, -829.036828547635)
        ]
    ]
}

cameras_intrinsic_params = [
    {
        'id': '54138969',
        'center': [630.814771, 377.307705],
        'focal_length': [1265.56639, 1268.81529],
        'radial_distortion': [-0.7806069408605851, 5.441217692399127, -23.568186296341604],
        'tangential_distortion': [-0.0023990326449613037, 0.003650403939447533],
        'res_w': 1280,
        'res_h': 720,
        'azimuth': 70, # Only used for visualization
    },
    {
        'id': '54138970',
        'center': [929.232643, 633.896163],
        'focal_length': [1643.67647, 1650.41390],
        'radial_distortion': [-0.12453684, 0.39429506, -0.72776789],
        'tangential_distortion': [0.0028723, -0.00632781],
        'res_w': 1920,
        'res_h': 1200,
        'azimuth': 70, # Only used for visualization
    }
]

all_cameras_extrinsic_params = {
    'G1': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
    'G2': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
    'G3': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
    'G4': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
    'G5': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
    'G6': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        },
        { # for camera 1
            'orientation': [-0.67986367, -0.62318688, 1.35579541], # rvec
            'translation': [59.60655896, -557.92403195, 4246.91242372] # tvec in mm
        }
    ],
}

vue_cameras_extrinsic_params = {
    'G1': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
    'G2': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
    'G3': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
    'G4': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
    'G5': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
    'G6': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ],
}

inference_camera_extrinsic_params = {
    'G1': [ 
        { # for camera 0
            'orientation': [-0.08974798, -0.03715601, 1.64284817], # rvec
            'translation': [356.2857915, -472.02995692, 4098.32075024] # tvec in mm
        }
    ]
}

class GolfDataset(MocapDataset):
    def __init__(self, path=None, remove_static_joints=False):
        super().__init__(fps=50, skeleton=golf_skeleton)
        
        ## self._cameras 表：每個subject要到每台相機的外參內參
        self._cameras = copy.deepcopy(inference_camera_extrinsic_params)
        for sub, cameras in self._cameras.items(): # for each subject's camera
            for i, cam in enumerate(cameras):
                cam.update(cameras_intrinsic_params[i])
                for k, v in cam.items():
                    if k not in ['id', 'res_w', 'res_h']:
                        cam[k] = np.array(v, dtype='float32')
                
                cam['mtx'] = np.array([
                    [cam['focal_length'][0], 0., cam['center'][0]],
                    [0., cam['focal_length'][1], cam['center'][1]],
                    [0., 0., 1.]
                ])
                cam['dist'] = np.array([
                    cam['radial_distortion'][0], cam['radial_distortion'][1], 
                    cam['tangential_distortion'][0], cam['tangential_distortion'][1], 
                    cam['radial_distortion'][2]
                ])
                cam['translation_mm'] = cam['translation']
                
                # # Normalize camera frame
                cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
                cam['focal_length'] = cam['focal_length']/cam['res_w']*2
                if 'translation' in cam:
                    cam['translation'] = cam['translation']/1000 # mm to meters
                
                # Add intrinsic parameters vector
                cam['intrinsic'] = np.concatenate((cam['focal_length'],
                                                   cam['center'],
                                                   cam['radial_distortion'],
                                                   cam['tangential_distortion']))
                cam['vicon_to_world_basis_dots'] = np.array(vicon_to_world_basis_dots[sub][i], dtype='float32')
                cam['world_to_vicon_basis_dots'] = np.array(world_to_vicon_basis_dots[sub][i], dtype='float32')
                cam['square_size'] = square_size

        # Load serialized dataset
        # data = np.load(path, allow_pickle=True)['positions_3d'].item()
        
        # self._data = {}
        # for subject, actions in data.items():
        #     self._data[subject] = {}
        #     for action_name, positions in actions.items():
                
        #         self._data[subject][action_name] = {
        #             'positions': positions,
        #             'cameras': self._cameras[subject],
        #         }
                
        # if remove_static_joints:
        #     # Bring the skeleton to 17 joints instead of the original 32
        #     self.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
            
        #     # Rewire shoulders to the correct parents
        #     self._skeleton._parents[11] = 8
        #     self._skeleton._parents[14] = 8
            
    def supports_semi_supervised(self):
        return True
   