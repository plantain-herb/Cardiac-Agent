patch_size = [64, 64, 64]
patch_size_inner = [48, 48, 48]
isotropy_spacing = [1, 1, 1]
data_pyramid_level = 3
data_pyramid_step = 2
trainner = dict(type='Trainner', runner_config=dict(type='EpochBasedRunner'))

model = dict(
    type="SegHeart_Network",
    backbone=dict(
        type="SCnet",
        in_ch=1,
        data_pyramid_level=data_pyramid_level,
        data_pyramid_step=data_pyramid_step,
        inner_backbone=[
            dict(type="ResUnet", in_ch=1, channels=32, stride=1),
            dict(type="ResSCUnet", in_ch=1, extra_in_ch=32, channels=32, stride=1),
            dict(type="ResSCUnet", in_ch=1, extra_in_ch=32, channels=32, stride=1),
        ],
    ),
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(
        type="HeartSegPyramid_Head",
        in_channels=32,
    ),
    pipeline=[
        dict(
            type="Augmentation3d",
            aug_parameters={
                # "rot_range_x": [-20, 20],
                # "rot_range_y": [-20, 20],
                # "rot_range_z": [-20, 20],
                # "scale_range_x": (0.9, 1.2),
                # "scale_range_y": (0.9, 1.2),
                # "scale_range_z": (0.9, 1.2),
                # "shift_range_x": (-0.1, 0.1),
                # "shift_range_y": (-0.1, 0.1),
                # "shift_range_z": (-0.1, 0.1),
                "elastic_alpha": [3.0, 3.0, 3.0],  # x,y,z
                "smooth_num": 4,
                "field_size": [17, 17, 17],  # x,y,z
                "size_o": patch_size,
                "itp_mode_dict": {"mask": "nearest"},
                "out_style": "crop",
            },
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
        type='HeartPyramid_Sample_Dataset',
        dst_list_file="/home/qutaiping/mr_heart_seg_thin/train/train_data/processed_data_second_stage2/train.lst",
        data_root="/home/qutaiping/mr_heart_seg_thin/train/train_data/processed_data_second_stage2",
        patch_size=patch_size,
        patch_size_inner=patch_size_inner,
        isotropy_spacing=isotropy_spacing,
        rotation_prob=0.95,
        rot_range=[15, 15, 15],
        spacing_range=0.05,
        shift_range=10,
        data_pyramid_level=3,
        data_pyramid_step=2,
        bg_sample_ratio=0.0,
        sample_frequent=10,
        constant_shift=20,
        whole_bright_aug=(0.5, 0.2, 0.2),  # (weight, bias)
        local_tp_bright_aug=(0.5, -0.2, 0.2),
    ),
)

optimizer = dict(type='Adam', lr=1e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, step=[10, 20], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

cudnn_benchmark = False
work_dir = './checkpoints/second_MMWHS'
gpus = 1
find_unused_parameters = True
total_epochs = 30
autoscale_lr = None
validate = False
launcher = 'pytorch'  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend='nccl')
log_level = 'INFO'
seed = None
deterministic = False
resume_from = None #"./checkpoints/second/m4.0.13_pretrain_4.0.3/latest.pth"
#load_from = "./checkpoints/secondm4.0.6_2/epoch_188.pth"
load_from = './checkpoints/second_MMWHS/latest.pth'
workflow = [('train', 1)]
