dataset_info = dict(
    dataset_name="GolfSwing",
    paper_info=dict(author="Ming-Han Lee and Yu-Chen Zhang and Kun-Ru Wu and Yu-Chee Tseng", title="GolfPose: From Regular Posture to Golf Swing Posture", container="International Conference on Pattern Recognition", year="2024", homepage="https://github.com/MingHanLee/GolfPose"),
    keypoint_info=dict(
        {
            0: dict(name="root", id=0, color=[51, 153, 255], type="lower", swap=""),
            1: dict(name="right_hip", id=1, color=[255, 128, 0], type="lower", swap="left_hip"),
            2: dict(name="right_knee", id=2, color=[255, 128, 0], type="lower", swap="left_knee"),
            3: dict(name="right_foot", id=3, color=[255, 128, 0], type="lower", swap="left_foot"),
            4: dict(name="left_hip", id=4, color=[0, 255, 0], type="lower", swap="right_hip"),
            5: dict(name="left_knee", id=5, color=[0, 255, 0], type="lower", swap="right_knee"),
            6: dict(name="left_foot", id=6, color=[0, 255, 0], type="lower", swap="right_foot"),
            7: dict(name="spine", id=7, color=[51, 153, 255], type="upper", swap=""),
            8: dict(name="thorax", id=8, color=[51, 153, 255], type="upper", swap=""),
            9: dict(name="neck_base", id=9, color=[51, 153, 255], type="upper", swap=""),
            10: dict(name="head", id=10, color=[51, 153, 255], type="upper", swap=""),
            11: dict(name="left_shoulder", id=11, color=[0, 255, 0], type="upper", swap="right_shoulder"),
            12: dict(name="left_elbow", id=12, color=[0, 255, 0], type="upper", swap="right_elbow"),
            13: dict(name="left_wrist", id=13, color=[0, 255, 0], type="upper", swap="right_wrist"),
            14: dict(name="right_shoulder", id=14, color=[255, 128, 0], type="upper", swap="left_shoulder"),
            15: dict(name="right_elbow", id=15, color=[255, 128, 0], type="upper", swap="left_elbow"),
            16: dict(name="right_wrist", id=16, color=[255, 128, 0], type="upper", swap="left_wrist"),
            17: dict(name="shaft", id=17, color=[255, 255, 255], type="upper", swap=""),
            18: dict(name="hosel", id=18, color=[255, 255, 255], type="lower", swap=""),
            19: dict(name="heel", id=19, color=[255, 255, 255], type="lower", swap=""),
            20: dict(name="toe_down", id=20, color=[255, 255, 255], type="lower", swap=""),
            21: dict(name="toe_up", id=21, color=[255, 255, 255], type="lower", swap=""),
        }
    ),
    skeleton_info=dict(
        {
            0: dict(link=("root", "left_hip"), id=0, color=[0, 255, 0]),
            1: dict(link=("left_hip", "left_knee"), id=1, color=[0, 255, 0]),
            2: dict(link=("left_knee", "left_foot"), id=2, color=[0, 255, 0]),
            3: dict(link=("root", "right_hip"), id=3, color=[255, 128, 0]),
            4: dict(link=("right_hip", "right_knee"), id=4, color=[255, 128, 0]),
            5: dict(link=("right_knee", "right_foot"), id=5, color=[255, 128, 0]),
            6: dict(link=("root", "spine"), id=6, color=[51, 153, 255]),
            7: dict(link=("spine", "thorax"), id=7, color=[51, 153, 255]),
            8: dict(link=("thorax", "neck_base"), id=8, color=[51, 153, 255]),
            9: dict(link=("neck_base", "head"), id=9, color=[51, 153, 255]),
            10: dict(link=("thorax", "left_shoulder"), id=10, color=[0, 255, 0]),
            11: dict(link=("left_shoulder", "left_elbow"), id=11, color=[0, 255, 0]),
            12: dict(link=("left_elbow", "left_wrist"), id=12, color=[0, 255, 0]),
            13: dict(link=("thorax", "right_shoulder"), id=13, color=[255, 128, 0]),
            14: dict(link=("right_shoulder", "right_elbow"), id=14, color=[255, 128, 0]),
            15: dict(link=("right_elbow", "right_wrist"), id=15, color=[255, 128, 0]),
            16: dict(link=("shaft", "hosel"), id=16, color=[255, 255, 255]),
            17: dict(link=("hosel", "heel"), id=17, color=[255, 255, 255]),
            18: dict(link=("heel", "toe_down"), id=18, color=[255, 255, 255]),
            19: dict(link=("toe_down", "toe_up"), id=19, color=[255, 255, 255]),
        }
    ),
    joint_weights=[1.0, 1.0, 1.2, 1.5, 1.0, 1.2, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.5, 1.0, 1.2, 1.5, 1.6, 1.9, 2.0, 2.0, 2.0],
    sigmas=[0.02] * 22,
)