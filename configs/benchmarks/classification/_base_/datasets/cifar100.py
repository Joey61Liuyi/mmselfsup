# dataset settings

dataset_type = 'mmcls.CIFAR100'
data_root = 'data/cifar100/'


img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding = 4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label']),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='train',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
    pin_memory=True,
)

val_dataloader = dict(
    batch_size=256,
    num_workers=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='val',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)

val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
