# dataset settings
dataset_type = 'mmcls.CIFAR100'
data_root = 'data/cifar100/'


view_pipeline1 = [
    dict(type='RandomResizedCrop', size=32),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomSolarize', prob=0.),
]

view_pipeline2 = [
    dict(type='RandomResizedCrop', size=32),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989)),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),
    dict(type='RandomSolarize', prob=0.),
]

# view_pipeline1 = [
#     dict(
#         type='RandomResizedCrop',
#         size=224,
#         interpolation='bicubic',
#         backend='pillow'),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='RandomApply',
#         transforms=[
#             dict(
#                 type='ColorJitter',
#                 brightness=0.4,
#                 contrast=0.4,
#                 saturation=0.2,
#                 hue=0.1)
#         ],
#         prob=0.8),
#     dict(
#         type='RandomGrayscale',
#         prob=0.2,
#         keep_channels=True,
#         channel_weights=(0.114, 0.587, 0.2989)),
#     dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
#     dict(type='RandomSolarize', prob=0.2),
# ]

img_norm_cfg = dict(
    mean=[129.304, 124.070, 112.434],
    std=[68.170, 65.392, 70.418],
    to_rgb=False)

train_pipeline = [
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='MultiView', num_views=[1,1], transforms=[view_pipeline1, view_pipeline2]),
    dict(type='PackSelfSupInputs', meta_keys=['img']),
]


# train_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=file_client_args),
#     dict(
#         type='MultiView',
#         num_views=[1, 1],
#         transforms=[view_pipeline1, view_pipeline2]),
#     dict(type='PackSelfSupInputs', meta_keys=['img_path'])
# ]

train_dataloader = dict(
    batch_size=256,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode = False,
        data_prefix='train/',
        pipeline=train_pipeline))