trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))

# win_level = 60
# win_width = 300
# other_win_level = 90
# other_win_width = 150
# other_win_level = (other_win_level - win_level + win_width / 2) / win_width
# other_win_width = other_win_width / win_width
# other_win_range = [other_win_level - other_win_width / 2, other_win_level + other_win_width / 2]

isotropy_spacing = 1
patch_size = [192, 192, 192]
patch_size_inner = [160, 160, 160]

model = dict(
    type="Seg_Network_Heart",
    backbone=dict(type="ResUnet", in_ch=1, channels=32, blocks=3),
    # other_win_range=other_win_range,
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(type="Seg_Head_Heart", in_channels=32, scale_factor=(2.0, 2.0, 2.0),),
    pipeline=[
        dict(
            type="Augmentation3d",
            aug_parameters={
                # "rot_range_x": [-20, 20],
                # "rot_range_y": [-20, 20],
                # "rot_range_z": [-20, 20],
                "scale_range_x": (0.9, 1.2),
                "scale_range_y": (0.9, 1.2),
                "scale_range_z": (0.9, 1.2),
                "shift_range_x": (-0.1, 0.1),
                "shift_range_y": (-0.1, 0.1),
                "shift_range_z": (-0.1, 0.1),
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
    imgs_per_gpu=1,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=2,),
    train=dict(
        type="Seg_Sample_Dataset",
        dst_list_file='/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_first_stage/train.lst',
        data_root="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_first_stage",
        isotropy_spacing=isotropy_spacing,
        patch_size=patch_size,
        patch_size_inner=patch_size_inner,
        rotation_prob=0.95,
        rot_range=[20, 20, 20],
        spacing_range=0.25,
        sample_frequent=10,
    ),
)

optimizer = dict(type="Adam", lr=1e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3, step=[2, 10], gamma=0.2)

checkpoint_config = dict(interval=1)  # save epoch

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "./checkpoints/first_MMWHS"
gpus = 4
find_unused_parameters = True
total_epochs = 20
autoscale_lr = None
validate = False
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
# resume_from = "checkpoints/liver_raw_0728/latest.pth"
resume_from = None
load_from = "./checkpoints/first_MMWHS/latest.pth"
workflow = [("train", 1)]
