
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing', # 使用余弦退火
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5         # 最终学习率降至初始值的 0.001%
)
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=1)