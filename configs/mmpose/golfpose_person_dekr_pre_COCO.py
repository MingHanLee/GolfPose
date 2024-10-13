default_scope = "mmpose"
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook", interval=10, save_best="coco/AP", rule="greater"
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
)
custom_hooks = [
    dict(type="SyncBuffersHook"),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=[
        dict(type="LocalVisBackend"),
    ],
    name="visualizer",
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False
backend_args = dict(backend="local")
train_cfg = dict(by_epoch=True, max_epochs=140, val_interval=10)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type="Adam", lr=0.001))
param_scheduler = [
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=140,
        milestones=[
            90,
            120,
        ],
        gamma=0.1,
        by_epoch=True,
    ),
]
auto_scale_lr = dict(base_batch_size=80)
codec = dict(
    type="SPR",
    input_size=(
        640,
        640,
    ),
    heatmap_size=(
        160,
        160,
    ),
    sigma=(
        4,
        2,
    ),
    minimal_diagonal_length=5.656854249492381,
    generate_keypoint_heatmaps=True,
    decode_max_instances=30,
)
model = dict(
    type="BottomupPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(
                    4,
                    4,
                ),
                num_channels=(
                    48,
                    96,
                ),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_channels=(
                    48,
                    96,
                    192,
                ),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_channels=(
                    48,
                    96,
                    192,
                    384,
                ),
                multiscale_output=True,
            ),
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth",
        ),
    ),
    neck=dict(type="FeatureMapProcessor", concat=True),
    head=dict(
        type="DEKRHead",
        in_channels=720,
        num_keypoints=17,
        num_heatmap_filters=48,
        heatmap_loss=dict(type="KeypointMSELoss", use_target_weight=True),
        displacement_loss=dict(
            type="SoftWeightSmoothL1Loss",
            use_target_weight=True,
            supervise_empty=False,
            beta=0.1111111111111111,
            loss_weight=0.002,
        ),
        decoder=dict(
            type="SPR",
            input_size=(
                640,
                640,
            ),
            heatmap_size=(
                160,
                160,
            ),
            sigma=(
                4,
                2,
            ),
            minimal_diagonal_length=5.656854249492381,
            generate_keypoint_heatmaps=True,
            decode_max_instances=30,
        ),
        # rescore_cfg=dict(
        #     in_channels=74,
        #     norm_indexes=(
        #         5,
        #         6,
        #     ),
        #     init_cfg=dict(
        #         type="Pretrained",
        #         checkpoint="https://download.openmmlab.com/mmpose/pretrain_models/kpt_rescore_coco-33d58c5c.pth",
        #     ),
        # ),
    ),
    test_cfg=dict(
        multiscale_test=False,
        flip_test=True,
        nms_dist_thr=0.05,
        shift_heatmap=True,
        align_corners=False,
    ),
)
find_unused_parameters = True

data_mode = "bottomup"
# data_root = "data/coco/"
data_root = "golfswing/"
train_pipeline = [
    dict(type="LoadImage"),
    dict(
        type="BottomupRandomAffine",
        input_size=(
            640,
            640,
        ),
    ),
    dict(type="RandomFlip", direction="horizontal"),
    dict(
        type="GenerateTarget",
        encoder=dict(
            type="SPR",
            input_size=(
                640,
                640,
            ),
            heatmap_size=(
                160,
                160,
            ),
            sigma=(
                4,
                2,
            ),
            minimal_diagonal_length=5.656854249492381,
            generate_keypoint_heatmaps=True,
            decode_max_instances=30,
        ),
    ),
    dict(type="BottomupGetHeatmapMask"),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(
        type="BottomupResize",
        input_size=(
            640,
            640,
        ),
        size_factor=32,
        resize_mode="expand",
    ),
    dict(
        type="PackPoseInputs",
        meta_keys=(
            "id",
            "img_id",
            "img_path",
            "crowd_index",
            "ori_shape",
            "img_shape",
            "input_size",
            "input_center",
            "input_scale",
            "flip",
            "flip_direction",
            "flip_indices",
            "raw_ann_info",
            "skeleton_links",
        ),
    ),
]

dataset_type = "CocoDataset"
# metainfo = dict(from_file="configs/_base_/datasets/golfswing_person.py")

train_dataloader = dict(
    batch_size=8,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode="bottomup",
        ann_file="coco/hscc_golf_person_2d_train.json",
        data_prefix=dict(img="images/"),
        # metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_mode="bottomup",
    ann_file="coco/hscc_golf_person_2d_test.json",
    data_prefix=dict(img="images/"),
    # metainfo=metainfo,
    test_mode=True,
    pipeline=val_pipeline,
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=val_dataset,
)
test_dataloader = val_dataloader

coco_to_h36m_converter = dict(
    type="KeypointConverter",
    num_keypoints=17,
    mapping=[
        (12, 1),
        (14, 2),
        (16, 3),
        (11, 4),
        (13, 5),
        (15, 6),
        (5, 11),
        (7, 12),
        (9, 13),
        (6, 14),
        (8, 15),
        (10, 16),
    ],
)
h36m_to_coco_converter = dict(
    type="KeypointConverter",
    num_keypoints=17,
    mapping=[
        (1, 12),
        (2, 14),
        (3, 16),
        (4, 11),
        (5, 13),
        (6, 15),
        (11, 5),
        (12, 7),
        (13, 9),
        (14, 6),
        (15, 8),
        (16, 10),
    ],
)
val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}coco/hscc_golf_person_2d_test.json",
    nms_mode="none",
    score_mode="keypoint",
    # pred_converter=coco_to_h36m_converter,
    # gt_converter=h36m_to_coco_converter
)

# COCO_partitions = dict(
#     left_arm=[5, 7, 9],
#     right_arm =[6, 8, 10],
#     left_leg=[11, 13, 15],
#     right_leg=[12, 14, 16],
#     all=range(17),
# )
# H36M_partitions = dict(
#     left_arm=[11, 12, 13],
#     right_arm =[14, 15, 16],
#     left_leg=[4, 5, 6],
#     right_leg=[1, 2, 3],
#     all=range(17),
# )

# val_evaluator = dict(
#     type="KeypointPartitionMetric",
#     metric=dict(
#         type="CocoMetric",
#         ann_file=f"{data_root}coco/hscc_golf_person_2d_test.json",
#         nms_mode="none",
#         score_mode="keypoint",
#         # pred_converter=coco_to_h36m_converter,
#         # gt_converter=h36m_to_coco_converter
#     ),
#     partitions=COCO_partitions,
# )

# val_evaluator = dict(
#     type="MultiDatasetEvaluator",
#     metrics=dict(
#         type="CocoMetric",
#         ann_file=f"{data_root}coco/hscc_golf_person_2d_test.json",
#         # use_area=False,
#         gt_converter=coco_to_h36m_converter,
#         # prefix="aic",
#     ),
#     datasets=val_dataset,
# )
test_evaluator = val_evaluator
