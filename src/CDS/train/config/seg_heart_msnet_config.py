patch_size_outer = (160, 160, 160)
patch_size_block = (64, 64, 64)
patch_size_block_inner = (64, 64, 64)
tgt_spacing_block = 1.0

# win_level = [60, 90]
# win_width = [300, 150]

trainner = dict(type='Trainner', runner_config=dict(type='EpochBasedRunner'))

model = dict(
    type='Ms_Heart_Network',
    backbone=dict(
        type='MSnet_Heart',
        outer_backbone=dict(type='ResBaseMSUnet_v1_Heart', in_ch=1, channels=32),
        inner_backbone=dict(type='ResMSUnet_Heart', in_ch=1, extra_in_ch=32, channels=32),
    ),
    apply_sync_batchnorm=True,
    head=dict(
        type='Ms_Heart_Head',
        in_channels=32,
        outer_scale_factor=2,
        inner_scale_factor=2
    ),
)

train_cfg = None
test_cfg = None


# 使用SampleDataLoader时使用
data = dict(
    imgs_per_gpu=1,
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
        type='MsDataset',
        dst_list_file="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_second_stage/train.lst",
        data_root="/home/taiping-qu/code/mr_heart_seg_thin/train/train_data/processed_data_second_stage",
        patch_size_outer=patch_size_outer,
        patch_size_block=patch_size_block,
        patch_size_block_inner=patch_size_block_inner,
        inner_sample_frequent=3,
        tgt_spacing_block=tgt_spacing_block,
        rotation_prob=0.95,
        rot_range=[20, 20, 20],
        tp_prob=1,
        sample_frequent=10,
    ),
    val=dict(
        type="Ms_Val_Dataset",
        dst_list_file="",
        data_root="",
        sample_frequent=1,
    )
)

optimizer = dict(type='Adam', lr=1e-4, weight_decay=5e-4)
optimizer_config = {}

lr_config = dict(policy='step', warmup='linear', warmup_iters=50, warmup_ratio=1.0 / 3, step=[3, 10, 20], gamma=0.2)

checkpoint_config = dict(interval=1)

log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

cudnn_benchmark = False
work_dir = './checkpoints/second'
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
load_from = './checkpoints/second/latest.pth'
workflow = [('train', 1)]
