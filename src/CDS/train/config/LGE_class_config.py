trainner = dict(type="Trainner", runner_config=dict(type="EpochBasedRunner"))
patch_size = [3, 256, 256]

# other_win_level = 60
# other_win_width = 300
# other_win_level = (other_win_level - win_level + win_width / 2) / win_width
# other_win_width = other_win_width / win_width
# other_win_range = [other_win_level - other_win_width / 2, other_win_level + other_win_width / 2]

# model = dict(
#     type="InfarClass_Network",
#     backbone=dict(type="Densenet36_SE_keepz_featmap", in_channels=1),
#     # other_win_range=other_win_range,
#     apply_sync_batchnorm=True,
#     head=dict(type="InfarClass_Head"),
#     pipeline=[
#     ],
# )
model = dict(
    type="LGEClassification_Network",
    backbone=dict(type="CNNTrans_multi2d", in_ch=3, channels=32, blocks=3),
    # other_win_range=other_win_range,
    apply_sync_batchnorm=True,
    head=dict(type="LGEClassification_Head"),
    # pipeline=[
    #     dict(
    #         type="Aug3dMini",
    #         aug_parameters=dict(
    #             rot_range_x=[-20, 20, 1.0],
    #             rot_range_y=[-20, 20, 1.0],
    #             rot_range_z=[-20, 20, 1.0],
    #             scale_range_x=[0.9, 1.1, 1.0],
    #             scale_range_y=[0.9, 1.1, 1.0],
    #             scale_range_z=[0.9, 1.1, 1.0],
    #             shift_range_x=[-0.1, 0.1, 1.0],
    #             shift_range_y=[-0.1, 0.1, 1.0],
    #             shift_range_z=[-0.1, 0.1, 1.0],
    #             flip_x=0.2,
    #             flip_y=0.2,
    #             flip_z=0.2,
    #             itp_mode_dict=dict(img="bilinear", mask="nearest"),
    #             ),
    #         )
    #     ],
)

train_cfg = None
test_cfg = None

# 使用SampleDataLoader时使用
data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=1,
    shuffle=True,
    drop_last=False,
    dataloader=dict(type="SampleDataLoader", source_batch_size=3, source_thread_count=1, source_prefetch_count=1,),
    train=dict(
        type="LGEClassificationPidReSampleDataset",
        root="/home/qutaiping/nas/processed_data_LGE_class",
        dst_list_file="/home/qutaiping/nas/processed_data_LGE_class/train.lst",
        patch_size=patch_size,
        rotation_prob=0.5,
        noise_prob=0.1,
        color_prob=0.6,
        rot_range=[5, 5, 5],
        shift_range=10,
        sample_frequent=1,
        whole_bright_aug=(1, 0.1, 0.1),

    ),
    val=dict(
        type="LGE_Cls_ReSampleDataset_Val",
        root="/home/qutaiping/nas/processed_data_LGE_class/val",
        dst_list_file="/home/qutaiping/nas/processed_data_LGE_class/val/validation.lst",
        patch_size=patch_size,
        rotation_prob=0.0,
        noise_prob=0.0,
        color_prob=0.0,
        rot_range=[5, 5, 5],
        shift_range=5,
        sample_frequent=1,
    ),

)

optimizer = dict(type="Adam", lr=5e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy="step", warmup="linear", warmup_iters=10, warmup_ratio=1.0 / 3, step=[10, 40], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=1, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")])

cudnn_benchmark = False
work_dir = "/home/qutaiping/nas/checkpoints/LGE_class_finetune"
gpus = 4
find_unused_parameters = True
total_epochs = 100
autoscale_lr = None
validate = True
launcher = "pytorch"  # ['none', 'pytorch', 'slurm', 'mpi']
dist_params = dict(backend="nccl")
log_level = "INFO"
seed = None
deterministic = False
resume_from = None # "/home/qutaiping/nas/checkpoints/BLCG_class_flow_newmove_refine/latest.pth"
load_from = "/home/qutaiping/nas/checkpoints/LGE_class_finetune/latest.pth"
workflow = [("train", 1)]
