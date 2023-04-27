import argparse
import copy
import datetime
import glob
import os
import sys

from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer

from modules.callbacks import KeypointsLogger, SetupCallback
from modules.helpers import *

# torch.set_float32_matmul_precision('high')
# os.environ["TOKENIZERS_PARALLELISM"] = "true"


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-c',
        '--config',
        nargs='*',
        metavar='base_config.yaml',
        default=list(),
    )
    parser.add_argument(
        '-t',
        '--train',
        type=str2bool,
        default=True,
        nargs='?',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything'
    )
    parser.add_argument(
        '-f',
        '--fast_dev_run',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '-d',
        '--devices',
        # nargs="+",
        # type=int,
        default='auto'
    )
    parser.add_argument(
        '-a',
        '--accelerator',
        default='cpu'
    )
    parser.add_argument(
        '-e',
        '--max_epochs',
        type=int,
        default=0,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None
    )
    parser.add_argument(
        "--name",
        default=None
    )
    parser.add_argument(
        "--postfix",
        type=str,
        default=""
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs"
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None
    )
    parser.add_argument(
        "--scale_lr",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--no_test",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1
    )

    return parser
                    

if __name__=='__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    sys.path.append(os.getcwd())
    
    parser = get_parser()
    opt = parser.parse_args()

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot fine {opt.resume}")
        else:
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.config = base_configs + opt.config
        nowname = logdir.split("/")[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    # random seed
    seed_everything(opt.seed)

    # model config
    configs = [OmegaConf.load(cfg) for cfg in opt.config]
    config = OmegaConf.merge(*configs)
    
    lightning_config = OmegaConf.create(vars(opt))
    
    # initiate data
    data = instantiate_from_config(config.data)
    data.setup()
    
    total_steps = (len(data.datasets["train"]) // data.batch_size) * opt.max_epochs
    
    # initiate model
    config.model.params.noise_config.params["total_steps"] = total_steps
    model = instantiate_from_config(config.model)

    # update learning rate
    batch_size = config.data.params.batch_size
    base_lr_rate = config.model.params.base_learning_rate
    if opt.scale_lr:
        model.learning_rate = batch_size * base_lr_rate
        print(f"[INFO] Setting learning rate to {model.learning_rate:.2e} = {batch_size} (batchsize) * {base_lr_rate:.2e} (base_lr)")
    
    if not(opt.fast_dev_run):
        # initiate logger
        opt.logger = TensorBoardLogger(save_dir=logdir, version=nowname)
        
        # initiate callbacks
        opt.callbacks = [instantiate_from_config(config.custom_callback[callback]) for callback in config.custom_callback.keys()]
        
        opt.callbacks.append(ModelCheckpoint(dirpath=ckptdir, filename="{epoch:06}", 
                                            monitor=model.monitor, save_top_k=3))
        
        opt.callbacks.append(SetupCallback(resume=opt.resume, now=now, logdir=logdir, ckptdir=ckptdir, 
                                        cfgdir=cfgdir, config=config, lightning_config=lightning_config))
        
        opt.callbacks.append(EarlyStopping(monitor=model.monitor, verbose=True, patience=50))

    # lightning trainer
    trainer = Trainer.from_argparse_args(opt)
    
    if opt.train:
        trainer.fit(model, data)
    if not opt.no_test:
        trainer.test(model, data)
    
    