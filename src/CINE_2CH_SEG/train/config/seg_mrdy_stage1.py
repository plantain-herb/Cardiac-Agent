trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))
isotropy_spacing = 1.4063
patch_size = [96, 192, 192]
patch_size_inner = [80, 160, 160]


model = dict(
    type="SegDY_Network_Heart",
    backbone=dict(type="ResUnet", in_ch=1, channels=32, blocks=3),
    # other_win_range=other_win_range,
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(type="Seg_Head_Heart", in_channels=32, scale_factor=(2.0, 2.0, 2.0),),
    pipeline=[
        dict(
            type="Augmentation3d",
            aug_parameters={
                "flip_x": 0.2,
                "flip_y": 0.2,
                "flip_z": 0.2,
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
    imgs_per_gpu=8,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=2,),
    train=dict(
        type="SegDY_Sample_Dataset",
        dst_list_file='/home/qutaiping/nas/processed_data/processed_2CH_dy_stage1/train.lst',
        data_root="/home/qutaiping/nas/processed_data/processed_2CH_dy_stage1",
        isotropy_spacing=isotropy_spacing,
        # win_level=win_level,
        # win_width=win_width,
        patch_size=patch_size,
        patch_size_inner=patch_size_inner,
        rotation_prob=0.99,
        rot_range=[20, 20, 20],
        spacing_range=0.25,
        sample_frequent=10,
    ),
)

optimizer = dict(type="Adam", lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3, step=[3, 15, 30], gamma=0.2)

checkpoint_config = dict(interval=1)  # save epoch

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "/home/qutaiping/nas/checkpoints/4CH_dy_first1_refine"
gpus = 1
find_unused_parameters = True
total_epochs = 100
autoscale_lr = None
validate = False
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
# resume_from = "checkpoints/liver_raw_0728/latest.pth"
resume_from = None
# load_from = "./checkpoints/first_dy2/latest.pth"
load_from = None
workflow = [("train", 1)]
