dataset_info = dict(
    dataset_name="GolfSwing",
    paper_info=dict(author="Ming-Han Lee and Yu-Chen Zhang and Kun-Ru Wu and Yu-Chee Tseng", title="GolfPose: From Regular Posture to Golf Swing Posture", container="International Conference on Pattern Recognition", year="2024", homepage="https://github.com/MingHanLee/GolfPose"),
    keypoint_info=dict(
        {
            0: dict(name="shaft", id=0, color=[255, 255, 255], type="upper", swap=""),
            1: dict(name="hosel", id=1, color=[255, 255, 255], type="lower", swap=""),
            2: dict(name="heel", id=2, color=[255, 255, 255], type="lower", swap=""),
            3: dict(name="toe_down", id=3, color=[255, 255, 255], type="lower", swap=""),
            4: dict(name="toe_up", id=4, color=[255, 255, 255], type="lower", swap=""),
        }
    ),
    skeleton_info=dict(
        {
            0: dict(link=("shaft", "hosel"), id=0, color=[255, 255, 255]),
            1: dict(link=("hosel", "heel"), id=1, color=[255, 255, 255]),
            2: dict(link=("heel", "toe_down"), id=2, color=[255, 255, 255]),
            3: dict(link=("toe_down", "toe_up"), id=3, color=[255, 255, 255]),
        }
    ),
    joint_weights=[1.6, 1.9, 2.0, 2.0, 2.0],
    sigmas=[0.02] * 5,
)