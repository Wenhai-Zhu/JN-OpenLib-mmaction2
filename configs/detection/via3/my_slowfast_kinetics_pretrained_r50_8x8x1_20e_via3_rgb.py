model = dict(
    type='FastRCNN',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=4,
        speed_ratio=4,
        channel_ratio=8,
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            spatial_strides=(1, 2, 2, 1)),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            spatial_strides=(1, 2, 2, 1))),
    roi_head=dict(
        type='Via3RoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            in_channels=2304,
            num_classes=7,
            topk=(1, 1),
            multilabel=True,
            dropout_ratio=0.5)
    ),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0,
            debug=False)),
    test_cfg=dict(rcnn=dict(action_thr=0.0)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='SampleVia3Frames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals', 'gt_bboxes', 'gt_labels']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key=['proposals', 'gt_bboxes', 'gt_labels'], stack=False)
        ]),
    dict(
        type='Collect',
        keys=['img', 'proposals', 'gt_bboxes', 'gt_labels'],
        meta_keys=[ 'original_shape', 'img_shape', 'flip_direction', 'img_norm_cfg']
    )
]
val_pipeline = [
    dict(type='SampleVia3Frames', clip_len=32, frame_interval=2),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_bgr=False),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='Rename', mapping=dict(imgs='img')),
    dict(type='ToTensor', keys=['img', 'proposals']),
    dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
    dict(
        type='Collect',
        keys=['img', 'proposals'],
        meta_keys=['img_shape'],
        nested=True)
]

dataset_type = 'VIA3Dataset'
train_images_root = 'data/Interaction/images/train'
train_annotations_root = 'data/Interaction/annotations/train'

test_images_root = 'data/Interaction/images/test'
test_annotations_root = 'data/Interaction/annotations/test'

train_seq1_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq1.json',
    proposal_file=train_annotations_root + '/seq1_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq2_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq2.json',
    proposal_file=train_annotations_root + '/seq2_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq3_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq3.json',
    proposal_file=train_annotations_root + '/seq3_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)


train_seq4_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq4.json',
    proposal_file=train_annotations_root + '/seq4_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)


train_seq5_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq5.json',
    proposal_file=train_annotations_root + '/seq5_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq6_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq6.json',
    proposal_file=train_annotations_root + '/seq6_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq7_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq7.json',
    proposal_file=train_annotations_root + '/seq7_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq8_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq8.json',
    proposal_file=train_annotations_root + '/seq8_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq9_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq9.json',
    proposal_file=train_annotations_root + '/seq9_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq10_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq10.json',
    proposal_file=train_annotations_root + '/seq10_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq11_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq11.json',
    proposal_file=train_annotations_root + '/seq11_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq12_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq12.json',
    proposal_file=train_annotations_root + '/seq12_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq13_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq13.json',
    proposal_file=train_annotations_root + '/seq13_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)


train_seq14_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq14.json',
    proposal_file=train_annotations_root + '/seq14_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)


train_seq15_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq15.json',
    proposal_file=train_annotations_root + '/seq15_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq16_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq16.json',
    proposal_file=train_annotations_root + '/seq16_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq17_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq17.json',
    proposal_file=train_annotations_root + '/seq17_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq18_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq18.json',
    proposal_file=train_annotations_root + '/seq18_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq19_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq19.json',
    proposal_file=train_annotations_root + '/seq19_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)

train_seq20_cfg = dict(
    type=dataset_type,
    data_prefix=train_images_root,
    ann_file=train_annotations_root +  '/seq20.json',
    proposal_file=train_annotations_root + '/seq20_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=train_pipeline)



test_seq5_cfg = dict(
    type=dataset_type,
    data_prefix=test_images_root,
    ann_file=test_annotations_root +  '/seq5.json',
    proposal_file=test_annotations_root + '/seq5_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=val_pipeline)

test_seq10_cfg = dict(
    type=dataset_type,
    data_prefix=test_images_root,
    ann_file=test_annotations_root +  '/seq10.json',
    proposal_file=test_annotations_root + '/seq10_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=val_pipeline)

test_seq15_cfg = dict(
    type=dataset_type,
    data_prefix=test_images_root,
    ann_file=test_annotations_root +  '/seq15.json',
    proposal_file=test_annotations_root + '/seq15_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=val_pipeline)

test_seq20_cfg = dict(
    type=dataset_type,
    data_prefix=test_images_root,
    ann_file=test_annotations_root +  '/seq20.json',
    proposal_file=test_annotations_root + '/seq20_proposal.json',
    # custom_classes=['None', 'handshake', 'point', 'push'],
    attribute='person',
    custom_classes=None,
    pipeline=val_pipeline)


data = dict(
    # videos_per_gpu=8,
    # workers_per_gpu=10,
    videos_per_gpu=4,
    workers_per_gpu=5,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='ConcatDataset',
        datasets=[
            train_seq1_cfg,
            train_seq2_cfg,
            train_seq3_cfg,
            train_seq4_cfg,
            #train_seq5_cfg,
            train_seq6_cfg,
            train_seq7_cfg,
            train_seq8_cfg,
            train_seq9_cfg,
            #train_seq10_cfg,
            train_seq11_cfg,
            train_seq12_cfg,
            train_seq13_cfg,
            train_seq14_cfg,
            #train_seq15_cfg,
            train_seq16_cfg,
            train_seq17_cfg,
            train_seq18_cfg,
            train_seq19_cfg,
            #train_seq20_cfg,
        ],
        separate_eval=True,
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            test_seq5_cfg,
            test_seq10_cfg,
            test_seq15_cfg,
            test_seq20_cfg,
        ],
        separate_eval=True,
    ),
    test = dict(
        type='ConcatDataset',
        datasets=[
            test_seq5_cfg,
            test_seq10_cfg,
            test_seq15_cfg,
            test_seq20_cfg,
        ],
        separate_eval=True,
    ),
)

optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=1e-05)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(
    policy='step',
    step=[10, 15],
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.1)
total_epochs = 20
#total_epochs = 500
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
#evaluation = dict(interval=1, save_best='0_mAP@0.5IOU')
evaluation = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
#work_dir = './work_dirs/ava/slowfast_kinetics_pretrained_r50_8x8x1_20e_ava_rgb'
load_from = 'https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r50_8x8x1_256e_kinetics400_rgb/slowfast_r50_8x8x1_256e_kinetics400_rgb_20200716-73547d2b.pth'
#resume_from = 'work_dirs/my_slowfast_kinetics_pretrained_r50_8x8x1_20e_via3_ rgb/latest.pth'
resume_from = None
find_unused_parameters = False
gpu_ids = range(0, 1)
omnisource = False
module_hooks = []
