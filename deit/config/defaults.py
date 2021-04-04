# Copyright (c) Facebook, Inc. and its affiliates.
from .config import CfgNode as CN
from yacs.config import _VALID_TYPES

_VALID_TYPES.add(type(None))

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# The version number, to upgrade from old configs to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_C.VERSION = 2

_C.batch_size = 64
_C.epochs = 300
_C.model = ""
# customize model
_C.model_cfg = CN()
_C.model_cfg.type = ""
_C.model_cfg.kwargs = CN(new_allowed=True)
_C.input_size = 224
_C.drop = 0.0
_C.drop_path = 0.1
_C.model_ema = True
_C.model_ema_decay = 0.99996
_C.model_ema_force_cpu = False
_C.opt = "adamw"
_C.opt_eps = 1e-8
_C.opt_betas = None
_C.clip_grad = None
_C.momentum = 0.9
_C.weight_decay = 0.05
_C.sched = "cosine"
_C.lr = 5e-4
_C.lr_noise = None
_C.lr_noise_pct = 0.67
_C.lr_noise_std = 1.0
_C.warmup_lr = 1e-6
_C.min_lr = 1e-5
_C.decay_epochs = 30
_C.warmup_epochs = 5
_C.cooldown_epochs = 10
_C.patience_epochs = 10
_C.decay_rate = 0.1
_C.color_jitter = 0.4
_C.aa = "rand-m9-mstd0.5-inc1"
_C.smoothing = 0.1
_C.train_interpolation = 'bicubic'
_C.repeated_aug = True
_C.reprob = 0.25
_C.remode = 'pixel'
_C.recount = 1
_C.resplit = False
_C.mixup = 0.8
_C.cutmix = 1.
_C.cutmix_minmax = None
_C.mixup_prob = 1.
_C.mixup_switch_prob = 0.5
_C.mixup_mode = 'batch'
_C.teacher_model = ""
_C.teacher_path = ""
_C.distillation_type = 'none'
_C.distillation_alpha = 0.5
_C.distillation_tau = 1.0
_C.finetune = ''
_C.data_path = 'data/imagenet/2012'
_C.data_set = 'IMNET'
_C.inat_category = 'name'
_C.output_dir = None
_C.device = 'cuda'
_C.seed = 0
_C.resume = None
_C.start_epoch = 0
_C.eval = False
_C.dist_eval = False
_C.num_workers = 10
_C.pin_mem = True
_C.world_size = 1
_C.dist_url = 'env://'
_C.wandb = False
_C.save_interval = 15
_C.auto_resume = True
