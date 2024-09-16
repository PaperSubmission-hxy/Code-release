import math
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin, NativeMixedPrecisionPlugin

from src.config.default import get_cfg_defaults
from src.utils.misc import get_rank_zero_only_logger, setup_gpus
from src.utils.profiler import build_profiler
from src.lightning.data import MultiSceneDataModule
from src.lightning.lightning_loftr import PL_LoFTR

loguru_logger = get_rank_zero_only_logger(loguru_logger)

import pynvml
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--finetune_cfg_path', default=None)
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--batch_size', type=int, default=4, help='batch_size per gpu')
    parser.add_argument(
        '--n_workers', type=int, default=4)
    parser.add_argument(
        '--pin_memory', type=lambda x: bool(strtobool(x)),
        nargs='?', default=True, help='whether loading data to pinned memory or not')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--disable_ckpt', action='store_true',
        help='disable checkpoint saving (useful for debugging).')
    parser.add_argument(
        '--profiler_name', type=str, default=None,
        help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')
    parser.add_argument(
        '--method', type=str, default='loftr', help='choose loftr method')
    parser.add_argument(
        '--finetune', action='store_true', default=False, help='use finetune scheduler')
    parser.add_argument(
        '--val_oninit', action='store_true', default=False, help='use finetune scheduler')
    parser.add_argument(
        '--debug', action='store_true', default=False, help='debug')
    parser.add_argument(
        '--resume_from_latest', action='store_true', default=False, help='restore training in cluster')
    parser.add_argument(
        '--resume_aim_version', type=int, default=None, help='restore training in cluster')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main():
    # parse arguments
    args = parse_args()
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    get_cfg_default = get_cfg_defaults

    config = get_cfg_default()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.finetune_cfg_path) if args.finetune_cfg_path is not None else None
    config.merge_from_file(args.data_cfg_path)
    
    if config.LOFTR.COARSE.ROPE:
        assert config.DATASET.NPE_NAME is not None
    if config.DATASET.NPE_NAME is not None:
        if config.DATASET.NPE_NAME == 'megadepth':
            if isinstance(config.DATASET.MGDPT_IMG_RESIZE, int):
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE, config.DATASET.MGDPT_IMG_RESIZE]
            else:
                assert config.DATASET.MGDPT_IMG_RESIZE[0] == config.DATASET.MGDPT_IMG_RESIZE[1]
                config.LOFTR.COARSE.NPE = [832, 832, config.DATASET.MGDPT_IMG_RESIZE[0], config.DATASET.MGDPT_IMG_RESIZE[0]]
    
    if args.method is not None:
        config.METHOD = args.method

        if args.method in ["ROMA_SELF_TRAIN"]:
            config.DATASET.READ_GRAY = False
            config.DATASET.MGDPT_DF = None
    print(config)

    pl.seed_everything(config.TRAINER.SEED)  # reproducibility
    
    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)
    
    if args.debug:
        import debugpy
        # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
        debugpy.listen(5986)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    if args.resume_from_latest:
        # Find latest ckpt:
        latest_ckpt_path = None
        version_id = 0
        if args.resume_aim_version is not None:
            latest_ckpt_path = Path(config.DATASET.TB_LOG_DIR) / args.exp_name / f"version_{args.resume_aim_version}" / 'checkpoints' / 'last.ckpt'
            version_id = int(args.resume_aim_version)
        else:
            latest_ckpt_dir = list((Path(config.DATASET.TB_LOG_DIR) / args.exp_name).glob('version_*'))
            sorted_ckpt_paths = sorted(latest_ckpt_dir, key=lambda x: int(x.stem.split('_')[1]))
            for sorted_ckpt_path in sorted_ckpt_paths[::-1]:
                ckpt_path = Path(sorted_ckpt_path) / 'checkpoints' / 'last.ckpt'
                version_id = int(Path(sorted_ckpt_path).stem.split("_")[1])
                if ckpt_path.exists():
                    latest_ckpt_path = ckpt_path
                    break

        pl.seed_everything(config.TRAINER.SEED + version_id) 
        if latest_ckpt_path is not None and latest_ckpt_path.exists():
            args.resume_from_checkpoint = str(latest_ckpt_path)
            args.ckpt_path = str(latest_ckpt_path)
    
    # lightning module
    profiler = build_profiler(args.profiler_name)
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, profiler=profiler)
    loguru_logger.info(f"LoFTR LightningModule initialized!")
    
    if args.finetune:
        params_to_train = ['matcher.coarse_matching.classifier.weight', 'matcher.coarse_matching.classifier.bias']
        for name, param in model.named_parameters():
            param.requires_grad = True if name in params_to_train else False
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # lightning data
    data_module = MultiSceneDataModule(args, config)
    loguru_logger.info(f"LoFTR DataModule initialized!")
    
    # TensorBoard Logger
    logger = TensorBoardLogger(save_dir=config.DATASET.TB_LOG_DIR, name=args.exp_name, default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    
    # Callbacks
    ckpt_callback = ModelCheckpoint(monitor='auc@10', verbose=True, save_top_k=12, mode='max',
                                    save_last=True,
                                    dirpath=str(ckpt_dir),
                                    filename='{epoch}-{auc@5:.3f}-{auc@10:.3f}-{auc@20:.3f}')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [lr_monitor]
    if not args.disable_ckpt:
        callbacks.append(ckpt_callback)
    
    # Lightning Trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        plugins=[DDPPlugin(find_unused_parameters=False,
                          num_nodes=args.num_nodes,
                          sync_batchnorm=config.TRAINER.WORLD_SIZE > 0), NativeMixedPrecisionPlugin()],
        # plugins=[NativeMixedPrecisionPlugin()],
        gradient_clip_val=config.TRAINER.GRADIENT_CLIPPING,
        callbacks=callbacks,
        logger=logger,
        sync_batchnorm=config.TRAINER.WORLD_SIZE > 0,
        replace_sampler_ddp=False,  # use custom sampler
        reload_dataloaders_every_epoch=False,  # avoid repeated samples!
        weights_summary='full',
        # precision=16 if args.pre==16 else 32,
        profiler=profiler)
    loguru_logger.info(f"Trainer initialized!")
    loguru_logger.info(f"Start training!")

    # # debugpy.breakpoint()
    # # print('break on this line')
    
    pynvml.nvmlInit()
    if args.val_oninit:
        trainer.validate(model, datamodule=data_module)
    trainer.fit(model, datamodule=data_module)
    pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
