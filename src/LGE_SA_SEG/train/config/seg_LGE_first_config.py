trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))

# win_level = 60
# win_width = 300
# other_win_level = 90
# other_win_width = 150
# other_win_level = (other_win_level - win_level + win_width / 2) / win_width
# other_win_width = other_win_width / win_width
# other_win_range = [other_win_level - other_win_width / 2, other_win_level + other_win_width / 2]

patch_size = [3, 256, 256]
patch_size_inner = [3, 224, 224]
model = dict(
    type="Seg_Network_Heart",
    backbone=dict(type="ResUnet2d", in_ch=3, channels=32, blocks=3),
    # other_win_range=other_win_range,
    apply_sync_batchnorm=True,  # 默认为False, True表示使用sync_batchnorm，只有分布式训练才可以使用
    head=dict(type="Seg_Head_Heart2d", in_channels=32, scale_factor=(2.0, 2.0),),
    pipeline=[
        dict(
            type="Aug3dMini",
            aug_parameters=dict(
                rot_range_x=[-10, 10, 1.0],
                rot_range_y=[-10, 10, 1.0],
                rot_range_z=[-10, 10, 1.0],
                scale_range_x=[0.9, 1.1, 1.0],
                scale_range_y=[0.9, 1.1, 1.0],
                scale_range_z=[0.9, 1.1, 1.0],
                shift_range_x=[-0.1, 0.1, 1.0],
                shift_range_y=[-0.1, 0.1, 1.0],
                shift_range_z=[-0.1, 0.1, 1.0],
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
    imgs_per_gpu=8,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=2,),
    train=dict(
        type="Seg_Sample_Dataset2d",
        dst_list_file='/home/qutaiping/nas/processed_data/processed_data_LGE_SA_first_stage/train.lst',
        data_root="/home/qutaiping/nas/processed_data/processed_data_LGE_SA_first_stage",
        patch_size=patch_size,
        patch_size_inner=patch_size_inner,
        rotation_prob=0.95,
        rot_range=[20, 20, 20],
        noise_prob=0.1,
        color_prob=0.6,
        spacing_range=0.25,
        sample_frequent=10,
    ),
)

optimizer = dict(type="Adam", lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3, step=[10, 30], gamma=0.2)

checkpoint_config = dict(interval=1)  # save epoch

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "/home/qutaiping/nas/checkpoints/first_LGE_seg_refine_agent"
gpus = 4
find_unused_parameters = True
total_epochs = 40
autoscale_lr = None
validate = False
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
# resume_from = "checkpoints/liver_raw_0728/latest.pth"
resume_from = None
load_from = "/home/qutaiping/nas/checkpoints/first_LGE_seg_refine/latest.pth"
workflow = [("train", 1)]
