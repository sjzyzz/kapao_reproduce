import argparse
from copy import deepcopy
import logging
import os.path as osp
from pathlib import Path
import time
import yaml
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

import _init_path # make sure lib dir can be found
from lib.general import set_logging, colorstr, check_img_size, one_cycle, init_seeds
from lib.yolo import Model
from lib.torch_utils import torch_distributed_zero_first, de_parallel, intersect_dicts, ModelEMA
from lib.downloads import attempt_download
from lib.loss import MyComputeLoss
from lib.datasets import create_dataloader

from utils.loggers import Loggers
# from utils.datasets import create_dataloader
from utils.general import strip_optimizer, check_file, increment_path, check_dataset
from utils.torch_utils import select_device
from utils.autoanchor import check_anchors
from utils.metrics import fitness
from utils.callbacks import Callbacks
from val import run

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

LOGGER = logging.getLogger(__name__)
LOGGING_LEVEL = logging.INFO if RANK in [-1, 0] else logging.WARN
LOGGER.setLevel(LOGGING_LEVEL)

def train(hyp, opt, device, callbacks=Callbacks()):
    # Just get some variable
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, val_scales, val_flips = Path(
        opt.save_dir
    ), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.val_scales, opt.val_flips

    val_flips = [None if f == -1 else f for f in val_flips]

    # weights directories
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    last, best = weights_dir / 'last.pt', weights_dir / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)
    LOGGER.info(
        colorstr('hyperparameters: ') +
        ', '.join(f'{k}={v}' for k, v in hyp.items())
    )

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # TODO: register actions, but idn why
        pass

    # config
    plots = not evolve
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(
        RANK
    ):  # other process wait until the master process already loaded the `data_dict`, and then they execute the following code
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])
    names = [
        'items'
    ] if single_cls and len(data_dict['names']) != 1 else data_dict['names']
    assert len(
        names
    ) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'
    # is_coco = data.endswith('coco.yaml') and nc == 80

    labels_dir = data_dict.get('labels', 'labels')
    kp_flip = data_dict.get('kp_flip')
    kp_bbox = data_dict.get('kp_bbox')
    num_coords = data_dict.get('num_coords', 0)

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)
        ckpt = torch.load(weights, map_location=device)
        model = Model(
            cfg or ckpt['model'].yaml,
            ch=3,
            nc=nc,
            anchors=hyp.get('anchors'),
            num_coords=num_coords
        ).to(device)
        exclude = ['anchor'
                  ] if (cfg or hyp.get('anchors')) and not resume else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)
        LOGGER.info(
            f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}'
        )
    else:
        model = Model(
            cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), num_coords=num_coords
        ).to(device)

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    # TODO: idn why this stuff
    hyp['weight_decay'] *= batch_size * accumulate / nbs
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    # TODO: idn why divid into multiple groups
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    else:
        # TODO: idn the usage of `nesterov`
        optimizer = SGD(
            g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True
        )

    optimizer.add_param_group(
        {
            'params': g1,
            'weight_decay': hyp['weight_decay'],
        }
    )
    optimizer.add_param_group({'params': g2})
    # Scheduler
    if opt.linear_lr:
        lf = lambda epoch: (1 - epoch / (epochs - 1)) * (1.0 - hyp['lrf']
                                                        ) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # EMA
    # TODO: what this fucking `ModelEMA` doing lol
    # NOTE: this dude just keep the Exponential Moving Average of all parameters of the model.
    #       and only master process keep this is enough. Or say you donot know how to use this dude lol
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    start_epoch, best_fitness = 0, 0

    # Image size
    # TODO: why use this `max` function
    gs = max(int(model.stride.max()), 32)
    nl = model.model[-1].nl
    imgsz = check_img_size(opt.img_size, gs, floor=gs * 2)

    # DP mode
    if cuda and RANK == -1 and 1 < torch.cuda.device_count():
        logging.warning(
            'DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
            'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.'
        )
        model = torch.nn.DataParallel(model)

    # SynvBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        # TODO: take a deeper look at this line
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        labels_dir,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=opt.cache,
        rect=opt.rect,
        rank=RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr('train: '),
        kp_flip=kp_flip,
        kp_bbox=kp_bbox,
    )
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())
    nb = len(train_loader)
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(
            val_path,
            labels_dir,
            imgsz,
            batch_size // WORLD_SIZE,
            gs, 
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=False,
            rank=-1,
            workers=workers,
            pad=0.5,
            prefix=colorstr('val: '),
            kp_flip=kp_flip,
            kp_bbox=kp_bbox,
        )[0]

        if not resume:
            if not opt.noautoanchor:
                check_anchors(
                    dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz,
                )
            model.half().float()
        
        callbacks.on_pretrain_routine_end()

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640)**2 * 3. / nl
    hyp['kp'] *= 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    # model.class_weights = labels_to_class_weights(dataset.labels,
    #                                               nc).to(device) * nc
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0, 0)
    # TODO: idn why
    scheduler.last_epoch = start_epoch - 1
    # scaler = amp.GradScaler(enable=cuda)
    # stopper = EarlyStopping(patience=opt.patience)
    compute_loss = MyComputeLoss(
        model, autobalance=False, num_coords=num_coords
    )
    LOGGER.info(
        f'Image sizes {imgsz} train. {imgsz} val\n'
        f'Using {train_loader.num_workers} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )

    for epoch in range(start_epoch, epochs):
        model.train()

        # if opt.image_weights:
        #     # TODO: what the hell
        #     class_weights = model.class_weights.cpu().numpy(
        #     ) * (1 - maps)**2 / nc
        #     image_weights = labels_to_image_weights(
        #         dataset.labels, nc=nc, class_weights=cw
        #     )
        #     dataset.indices = random.choices(
        #         range(dataset.n), weights=image_weights, k=dataset.n
        #     )

        mloss = torch.zeros(4, device=device)
        # TODO: idn what is going on here
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(
            ('\n' + '%10s' * 8) % (
                'Epoch', 'gpu_men', 'box', 'obj', 'cls', 'kps', 'labels',
                'img_size'
            )
        )
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:
            ni = i + nb * epoch
            # TODO: idn the usage of `non_blocking`
            # TODO: i think the operation should be taken to `dataset`
            imgs = imgs.to(device, non_blocking=True).float() / 255.0

            # Warmup
            # if ni <= nw:
            #     pass
            # Multi-scale
            # if opt.multi_scale:
            #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs
            #     sf = sz / max(imgs.shape[2:])
            #     if sf != 1:
            #         # stretched to greatest-stride-multiple
            #         ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]
            #         imgs = nn.functional.interpolate(
            #             imgs, size=ns, mode='bilinear', align_corners=False
            #         )

            # Forward
            # TODO: idn why this statement
            # with amp.autocast(enabled=cuda):
            # get your fucking model lol
            pred = model(imgs)
            loss, loss_items = compute_loss(
                pred, targets.to(device)
            )  # loss scaled by batch_size
            # TODO: idn why those multiply
            if RANK != -1:
                loss *= WORLD_SIZE
            if opt.quad:
                loss *= 4.

            # Backward
            # TODO: idn why this statement
            # scaler.scale(loss).backward()
            loss.backward()

            # Optimize
            # TODO: idn why this warp
            # if accumulate <= ni - last_opt_step:
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                # TODO: idn the meaning of `.3g`
                men = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                pbar.set_description(
                    ('%10s' * 2 + '%10.4g' * 6) % (
                        f'{epoch}/{epochs - 1}', men, *mloss, targets.shape[0],
                        imgs.shape[-1]
                    )
                )
                # callbacks.on_train_batch_end(
                #     ni, model, imgs, targets, paths, plots, opt.sync_bn
                # )

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups[:3]]
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)
            ema.updata_attr(
                model,
                include=[
                    'yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights',
                ]
            )
            # TODO: I just ignore the early stop here
            final_epoch = (epoch + 1 == epochs)
            if not noval or final_epoch:
                results, maps, _ = run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE,
                    imgsz=imgsz,
                    conf_thres=0.01,
                    model=ema.ema,
                    dataloader=val_loader,
                    compute_loss=compute_loss,
                    scales=val_scales,
                    flips=val_flips,
                )
            
            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.on_model_save(
                    last, epoch, final_epoch, best_fitness, fi
                )

    # end training
    if RANK in [-1, 0]:
        LOGGER.info(
            f'\n{epoch-start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours'
        )
        if not evolve:
            # TODO: what the hell is this stuff
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)
        # callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # GLOBAL
    # TODO: wth are these argument
    # if there is no --resume, then set this value to False, if there is only --resume, set this value to True,
    # and if there are --resume val, set this value to val
    parser.add_argument(
        '--resume',
        nargs='?',
        const=True,
        default=False,
        help='resume most recent training'
    )
    parser.add_argument(
        '--evolve',
        type=int,
        nargs='?',
        const=300,
        help='evolve hyperparameters for x generations'
    )
    # MODEL
    parser.add_argument('--hyp', type=str, default='cfg/hyp/hyp.kp-p6.yaml')
    parser.add_argument(
        '--weights',
        type=str,
        default='kapao_s_coco.pt',
        help='initial weights path'
    )
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument(
        '--single-cls',
        action='store_true',
        help='train multi-class data as single-class'
    )
    parser.add_argument(
        '--noautoanchor',
        action='store_true',
        help='disable autoachor check'
    )
    # DATASET
    parser.add_argument(
        '--data',
        type=str,
        default='data/coco-kp.yaml',
        help='dataset.yaml path'
    )
    parser.add_argument(
        '--img-size', type=int, default=1280, help='train, val image size'
    )
    parser.add_argument(
        '--rect', action='store_true', help='rectangular training'
    )
    parser.add_argument(
        '--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"'
    )
    parser.add_argument(
        '--quad', action='store_true', help='quad dataloader'
    )
    parser.add_argument(
        '--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon'
    )
    # TRAIN
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='total batch size for all GPUs'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='maximum number of dataloader workers'
    )
    parser.add_argument(
        '--image-weights',
        action='store_true',
        help='use weighted image selection for training'
    )
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument(
        '--noval', action='store_true', help='only validate final epoch'
    )
    parser.add_argument(
        '--freeze',
        type=int,
        default=0,
        help='Number of layers to freeze. backbone=10, all=24'
    )
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # OPTIMIZER
    parser.add_argument(
        '--adam',
        action='store_true',
        help='use torch.optim.Adam() optimizer'
    )
    parser.add_argument(
        '--linear-lr',
        action='store_true',
        help='linear LR'
    )
    # VALIDATE
    parser.add_argument('--val-scales', type=float, nargs='+', default=[1])
    parser.add_argument('--val-flips', type=int, nargs='+', default=[-1])
    # SAVE
    parser.add_argument(
        '--project', default='runs/train', help='save to project/name'
    )
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='existing project/name ok, do not increment'
    )
    parser.add_argument(
        '--nosave', action='store_true', help='only save final checkpoint'
    )

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(
            f"{colorstr('train: ')}{', '.join(f'{k}={v}' for k, v in vars(opt).items())}"
        )

    if opt.resume and not opt.evolve:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert osp.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))
        opt.cfg, opt.weights, opt.resume = '', ckpt, True
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(
            opt.cfg
        ), check_file(opt.hyp)
        assert len(opt.cfg) or len(
            opt.weights
        ), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(
            increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
        )

        device = select_device(opt.device, batch_size=opt.batch_size)
        if LOCAL_RANK != -1:
            from datetime import timedelta
            assert LOCAL_RANK < torch.cuda.device_count(
            ), 'insufficient CUDA devices for DDP command'
            assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
            assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
            assert not opt.evolve, '--evolve argument is not compatible with DDP training'
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device('cuda', LOCAL_RANK)
            dist.init_process_group(
                backend='nccl' if dist.is_nccl_available() else 'gloo'
            )

        if not opt.evolve:
            train(opt.hyp, opt, device)
            if 1 < WORLD_SIZE and RANK == 0:
                _ = [
                    print('Destroying process group... ', end=''),
                    dist.destroy_process_group(),
                    print('Done.')
                ]


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)