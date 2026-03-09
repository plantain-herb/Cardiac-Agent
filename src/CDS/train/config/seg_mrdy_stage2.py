patch_size = [64, 64, 64]
patch_size_inner = [48, 48, 48]
isotropy_spacing = [0.8, 0.8, 0.8]
data_pyramid_level = 2
data_pyramid_step = 2
trainner = dict(type='Trainner', runner_config=dict(type='EpochBasedRunner'))
# win_level = 350
# win_width = 700
# other_win_level = 170
# other_win_width = 350
# win_level = 2000
# win_width = 4000
# other_win_level = 1000
# other_win_width = 2000

# other_win_level = (other_win_level - win_level + win_width / 2) / win_width
# other_win_width = other_win_width / win_width
# other_win_range = [
#     other_win_level - other_win_width / 2,
#     other_win_level + other_win_width / 2,
# ]

model = dict(
    type="SegDY_Network",
    backbone=dict(
        type="SCnet_2_1c",
        in_ch=1,
        data_pyramid_level=data_pyramid_level,
        data_pyramid_step=data_pyramid_step,
        inner_backbone=[
            dict(type="ResUnet", in_ch=1, channels=32, stride=1),
            dict(type="ResSCUnet", in_ch=1, extra_in_ch=32, channels=32, stride=1),
        ],
    ),
    # other_win_range=other_win_range,
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(
        type="DYSegPyramid_Head",
        in_channels=32,
    ),

    pipeline=[
        dict(
            type="Aug3dMini",
            aug_parameters=dict(
                rot_range_x=[-20, 20, 1.0],
                rot_range_y=[-20, 20, 1.0],
                rot_range_z=[-20, 20, 1.0],
                scale_range_x=[0.9, 1.1, 1.0],
                scale_range_y=[0.9, 1.1, 1.0],
                scale_range_z=[0.9, 1.1, 1.0],
                shift_range_x=[-0.1, 0.1, 1.0],
                shift_range_y=[-0.1, 0.1, 1.0],
                shift_range_z=[-0.3, 0.3, 0.2],
                flip_x=0.2,
                flip_y=0.2,
                flip_z=0.2,
                itp_mode_dict=dict(img="bilinear", mask="nearest"),
            ),
        )
    ],
)

train_cfg = None
test_cfg = None


# 使用SampleDataLoader时使用
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(
        type='SampleDataLoader',
        source_batch_size=3,
        source_thread_count=2,
        source_prefetch_count=3,
    ),
    train=dict(
        type='DYPyramid_Sample_Dataset',
        dst_list_file='/home/qutaiping/nas/processed_data_dy_stage2/train.lst',
        data_root="/home/qutaiping/nas/processed_data_dy_stage2",
        patch_size=patch_size,
        patch_size_inner=patch_size_inner,
        # win_level=win_level,
        # win_width=win_width,
        isotropy_spacing=isotropy_spacing,
        rotation_prob=0.5,
        rot_range=[5, 5, 5],
        spacing_range=0.05,
        shift_range=10,
        data_pyramid_level=2,
        data_pyramid_step=2,
        bg_sample_ratio=0.0,
        sample_frequent=4,
        constant_shift=20,
        whole_bright_aug=(0.8, 0.2, 0.2),  # (weight, bias)
        local_tp_bright_aug=(0.8, -0.1, 0.3),
    )
)

optimizer = dict(type='Adam', lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, step=[8, 24, 55], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

cudnn_benchmark = False
# work_dir = './checkpoints/second_dy_sp075'
work_dir = './checkpoints/second_dy_newnor_finetune64_08'
# work_dir = './checkpoints/second_dy_sp075_win350'
# work_dir = './checkpoints/second_dy_sp025_win350'
gpus = 1
find_unused_parameters = True
total_epochs = 70
autoscale_lr = None
validate = False
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None # './checkpoints/second_dy_newnor_finetune64/latest.pth'
#load_from = "./checkpoints/secondm4.0.6_2/epoch_188.pth"
load_from = './checkpoints/second_dy_newnor_finetune64/latest.pth'
workflow = [('train', 1)]
