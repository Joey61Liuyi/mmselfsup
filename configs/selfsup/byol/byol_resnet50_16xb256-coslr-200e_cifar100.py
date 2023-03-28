_base_ = [
    '../_base_/models/byol.py',
    '../_base_/datasets/cifar100_byol.py',
    '../_base_/schedules/lars_coslr-200e_cifar100.py',
    '../_base_/default_runtime.py',
]

# optimizer
optimizer = dict(type='LARS', lr=4.8, momentum=0.9, weight_decay=1e-6)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'bn': dict(decay_mult=0, lars_exclude=True),
            'bias': dict(decay_mult=0, lars_exclude=True),
            # bn layer in ResNet block downsample module
            'downsample.1': dict(decay_mult=0, lars_exclude=True),
        }),
)

# runtime settings
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))
