import os
import argparse
import shutil
import time
from collections import OrderedDict
from os import path
import tensorboardX as tensorboard
import torch
import torch.optim as optim
import torch.utils.data as data
from torch import distributed
from inplace_abn import ABN

from dvt.config import load_config

from dvt.data.dataset import BEVKitti360Dataset, BEVTransform, BEVNuScenesDataset
from dvt.data.misc import iss_collate_fn
from dvt.data.sampler import DistributedARBatchSampler

from dvt.modules.ms_transformer import MultiScaleTransformerVF
from dvt.modules.heads import FPNSemanticHeadDPC as FPNSemanticHeadDPC, FPNMaskHead, RPNHead

from dvt.models.backbone_edet.efficientdet import EfficientDet
from dvt.models.dvt import DvtNet, NETWORK_INPUTS

from dvt.algos.transformer import TransformerVFAlgo, TransformerVFLoss, TransformerRegionSupervisionLoss
from dvt.algos.fpn import InstanceSegAlgoFPN, RPNAlgoFPN
from dvt.algos.instance_seg import PredictionGenerator as MskPredictionGenerator, InstanceSegLoss
from dvt.algos.rpn import AnchorMatcher, ProposalGenerator, RPNLoss
from dvt.algos.detection import PredictionGenerator as BbxPredictionGenerator, DetectionLoss, ProposalMatcher
from dvt.algos.semantic_seg import SemanticSegLoss, SemanticSegAlgo
from dvt.algos.po_fusion import PanopticLoss, PanopticFusionAlgo

from dvt.utils import logging
from dvt.utils.meters import AverageMeter, ConfusionMatrixMeter, ConstantMeter
from dvt.utils.misc import config_to_string, scheduler_from_config, norm_act_from_config, all_reduce_losses
from dvt.utils.parallel import DistributedDataParallel
from dvt.utils.snapshot import save_snapshot, resume_from_snapshot, pre_train_from_snapshots
from dvt.utils.sequence import pad_packed_images
from dvt.utils.panoptic import compute_panoptic_test_metrics, panoptic_post_processing, get_scores


parser = argparse.ArgumentParser(description="Panoptic BEV Training Script")
parser.add_argument("--local_rank", required=True, type=int)
parser.add_argument("--run_name", required=True, type=str, help="Name of the run for creating the folders")
parser.add_argument("--project_root_dir", required=True, type=str, help="The root directory of the project")
parser.add_argument("--seam_root_dir", required=True, type=str, help="Seamless dataset directory")
parser.add_argument("--dataset_root_dir", required=True, type=str, help="Kitti360/nuScenes directory")
parser.add_argument("--mode", required=True, type=str, help="'train' the model or 'test' the model")
parser.add_argument("--train_dataset", type=str, choices=['Kitti360', 'nuScenes'], help="Dataset for training")
parser.add_argument("--val_dataset", type=str, choices=['Kitti360', 'nuScenes'], help="Dataset for validation")
parser.add_argument("--resume", metavar="FILE", type=str, help="Resume training from given file", nargs="?")
parser.add_argument("--eval", action="store_true", help="Do a single validation run")
parser.add_argument("--pre_train", type=str, nargs="*",
                    help="Start from the given pre-trained snapshots, overwriting each with the next one in the list. "
                         "Snapshots can be given in the format '{module_name}:{path}', where '{module_name} is one of "
                         "'body', 'rpn_head', 'roi_head' or 'sem_head'. In that case only that part of the network "
                         "will be loaded from the snapshot")
parser.add_argument("--config", required=True, type=str, help="Path to configuration file")
parser.add_argument("--debug", type=bool, default=False, help="Should the program run in 'debug' mode?")
parser.add_argument("--freeze_modules", nargs='+', default=[], help="The modules to freeze. Default is empty")

def log_info(msg, *args, **kwargs):
    if "debug" in kwargs.keys():
        print(msg % args)
    else:
        if distributed.get_rank() == 0:
            logging.get_logger().info(msg, *args, **kwargs)


def log_miou(label, miou, classes):
    logger = logging.get_logger()
    padding = max(len(cls) for cls in classes)

    logger.info("---------------- {} ----------------".format(label))
    for miou_i, class_i in zip(miou, classes):
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(class_i, miou_i.item()))


def log_scores(label, scores):
    logger = logging.get_logger()
    padding = max(len(cls) for cls in scores.keys())

    logger.info("---------------- {} ----------------".format(label))
    for score_label, score_value in scores.items():
        logger.info(("{:>" + str(padding) + "} : {:.5f}").format(score_label, score_value.item()))


def make_config(args, config_file):
    log_info("Loading configuration from %s", config_file, debug=args.debug)
    conf = load_config(config_file)

    log_info("\n%s", config_to_string(conf), debug=args.debug)
    return conf


def create_run_directories(args, rank):
    root_dir = args.project_root_dir
    experiment_dir = os.path.join(root_dir, "experiments")
    if args.mode == "train":
        run_dir = os.path.join(experiment_dir, "bev_train_{}".format(args.run_name))
    elif args.mode == "test":
        run_dir = os.path.join(experiment_dir, "bev_test_{}".format(args.run_name))
    else:
        raise RuntimeError("Invalid choice. --mode must be either 'train' or 'test'")
    saved_models_dir = os.path.join(run_dir, "saved_models")
    log_dir = os.path.join(run_dir, "logs")
    config_file = os.path.join(run_dir, args.config)

    # Create the directory
    if rank == 0 and (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
    if rank == 0:
        assert not os.path.exists(run_dir), "Run folder already found! Delete it to reuse the run name."

    if rank == 0:
        os.mkdir(run_dir)
        os.mkdir(saved_models_dir)
        os.mkdir(log_dir)

    # Copy the config file into the folder
    config_path = os.path.join(experiment_dir, "config", args.config)
    if rank == 0:
        shutil.copyfile(config_path, config_file)

    return log_dir, saved_models_dir, config_path


def make_dataloader(args, config, rank, world_size):
    dl_config = config['dataloader']

    log_info("Creating train dataloader for {} dataset".format(args.train_dataset), debug=args.debug)
    log_info("Creating val dataloader for {} dataset".format(args.val_dataset), debug=args.debug)

    # Train dataloaders
    train_tf = BEVTransform(shortest_size=dl_config.getint("shortest_size"),
                            longest_max_size=dl_config.getint("longest_max_size"),
                            rgb_mean=dl_config.getstruct("rgb_mean"),
                            rgb_std=dl_config.getstruct("rgb_std"),
                            front_resize=dl_config.getstruct("front_resize"),
                            bev_crop=dl_config.getstruct("bev_crop"),
                            scale=dl_config.getstruct("scale"),
                            random_flip=dl_config.getboolean("random_flip"),
                            random_brightness=dl_config.getstruct("random_brightness"),
                            random_contrast=dl_config.getstruct("random_contrast"),
                            random_saturation=dl_config.getstruct("random_saturation"),
                            random_hue=dl_config.getstruct("random_hue"))

    if args.train_dataset == "Kitti360":
        train_db = BEVKitti360Dataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                      split_name=dl_config['train_set'], transform=train_tf)
    elif args.train_dataset == "nuScenes":
        train_db = BEVNuScenesDataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                      split_name=dl_config['train_set'], transform=train_tf)

    if not args.debug:
        train_sampler = DistributedARBatchSampler(train_db, dl_config.getint('train_batch_size'), world_size, rank, True)
        train_dl = torch.utils.data.DataLoader(train_db,
                                               batch_sampler=train_sampler,
                                               collate_fn=iss_collate_fn,
                                               pin_memory=True,
                                               num_workers=dl_config.getint("train_workers"))
    else:
        train_dl = torch.utils.data.DataLoader(train_db,
                                               batch_size=dl_config.getint('train_batch_size'),
                                               collate_fn=iss_collate_fn,
                                               pin_memory=True,
                                               num_workers=dl_config.getint("train_workers"))

    # Validation datalaader
    val_tf = BEVTransform(shortest_size=dl_config.getint("shortest_size"),
                          longest_max_size=dl_config.getint("longest_max_size"),
                          rgb_mean=dl_config.getstruct("rgb_mean"),
                          rgb_std=dl_config.getstruct("rgb_std"),
                          front_resize=dl_config.getstruct("front_resize"),
                          bev_crop=dl_config.getstruct("bev_crop"))

    if args.val_dataset == "Kitti360":
        val_db = BEVKitti360Dataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                    split_name=dl_config['val_set'], transform=val_tf)
    elif args.val_dataset == "nuScenes":
        val_db = BEVNuScenesDataset(seam_root_dir=args.seam_root_dir, dataset_root_dir=args.dataset_root_dir,
                                    split_name=dl_config['val_set'], transform=val_tf)

    if not args.debug:
        val_sampler = DistributedARBatchSampler(val_db, dl_config.getint("val_batch_size"), world_size, rank, False)
        val_dl = torch.utils.data.DataLoader(val_db,
                                             batch_sampler=val_sampler,
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))
    else:
        val_dl = torch.utils.data.DataLoader(val_db,
                                             batch_size=dl_config.getint("val_batch_size"),
                                             collate_fn=iss_collate_fn,
                                             pin_memory=True,
                                             num_workers=dl_config.getint("val_workers"))

    return train_dl, val_dl


def make_model(args, config, num_thing, num_stuff):
    base_config = config["base"]
    fpn_config = config["fpn"]
    transformer_config = config['transformer']
    rpn_config = config['rpn']
    roi_config = config['roi']
    sem_config = config['sem']
    cam_config = config['cameras']
    dl_config = config['dataloader']

    num_stuff = num_stuff
    num_thing = num_thing
    classes = {"total": num_thing + num_stuff, "stuff": num_stuff, "thing": num_thing}

    # BN + activation
    if not args.debug:
        norm_act_static, norm_act_dynamic = norm_act_from_config(base_config)
    else:
        norm_act_static, norm_act_dynamic = ABN, ABN


    # Create the backbone
    # from dvt.models.backbone_edet.resnet import ResNetBackbone

    model_compount_coeff = int(base_config["base"][-1])
    model_name = "efficientdet-d{}".format(model_compount_coeff)
    # model_compount_coeff = 152
    # model_name = "resnet{}".format(model_compount_coeff)
    log_info("Creating backbone model %s", base_config["base"], debug=args.debug)
    # body = ResNetBackbone()
    body = EfficientDet(compound_coef=model_compount_coeff)
    ignore_layers = ['bifpn.0.p5_to_p6', 'bifpn.0.p6_to_p7', 'bifpn.0.p3_down_channel', "bifpn.0.p4_down_channel",
                   "bifpn.0.p5_down_channel", "bifpn.0.p6_down_channel"]
    body = EfficientDet.from_pretrained(body, model_name, ignore_layers=ignore_layers)

    # The transformer operates only on a single scale
    extrinsics = cam_config.getstruct('extrinsics')
    bev_params = cam_config.getstruct('bev_params')
    tfm_scales = transformer_config.getstruct("tfm_scales")

    bev_transformer = MultiScaleTransformerVF(transformer_config.getint("in_channels"),
                                              transformer_config.getint("tfm_channels"),
                                              transformer_config.getint("bev_ms_channels"),
                                              extrinsics, bev_params,
                                              H_in=dl_config.getstruct("front_resize")[0] * dl_config.getfloat("scale"),
                                              W_in=dl_config.getstruct('front_resize')[1] * dl_config.getfloat('scale'),
                                              Z_out=dl_config.getstruct("bev_crop")[1] * dl_config.getfloat('scale'),
                                              W_out=dl_config.getstruct('bev_crop')[0] * dl_config.getfloat('scale'),
                                              tfm_scales=tfm_scales,
                                              use_init_theta=transformer_config['use_init_theta'],
                                              norm_act=norm_act_static)

    vf_loss = TransformerVFLoss()
    region_supervision_loss = TransformerRegionSupervisionLoss()
    transformer_algo = TransformerVFAlgo(vf_loss, region_supervision_loss)

    # Create RPN
    proposal_generator = ProposalGenerator(rpn_config.getfloat("nms_threshold"),
                                           rpn_config.getint("num_pre_nms_train"),
                                           rpn_config.getint("num_post_nms_train"),
                                           rpn_config.getint("num_pre_nms_val"),
                                           rpn_config.getint("num_post_nms_val"),
                                           rpn_config.getint("min_size"))
    anchor_matcher = AnchorMatcher(rpn_config.getint("num_samples"),
                                   rpn_config.getfloat("pos_ratio"),
                                   rpn_config.getfloat("pos_threshold"),
                                   rpn_config.getfloat("neg_threshold"),
                                   rpn_config.getfloat("void_threshold"))
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    anchor_scales = [int(scale) for scale in rpn_config.getstruct('anchor_scale')]
    anchor_ratios = [float(ratio) for ratio in rpn_config.getstruct('anchor_ratios')]
    rpn_algo = RPNAlgoFPN(proposal_generator, anchor_matcher, rpn_loss, anchor_scales, anchor_ratios,
                          fpn_config.getstruct("out_strides"), rpn_config.getint("fpn_min_level"),
                          rpn_config.getint("fpn_levels"))
    rpn_head = RPNHead(transformer_config.getint("bev_ms_channels"), int(len(anchor_scales) * len(anchor_ratios)), 1,
                       rpn_config.getint("hidden_channels"), norm_act_dynamic)

    # Create instance segmentation network
    bbx_prediction_generator = BbxPredictionGenerator(roi_config.getfloat("nms_threshold"),
                                                      roi_config.getfloat("score_threshold"),
                                                      roi_config.getint("max_predictions"),
                                                      dataset_name=args.train_dataset)
    msk_prediction_generator = MskPredictionGenerator()
    roi_size = roi_config.getstruct("roi_size")
    proposal_matcher = ProposalMatcher(classes,
                                       roi_config.getint("num_samples"),
                                       roi_config.getfloat("pos_ratio"),
                                       roi_config.getfloat("pos_threshold"),
                                       roi_config.getfloat("neg_threshold_hi"),
                                       roi_config.getfloat("neg_threshold_lo"),
                                       roi_config.getfloat("void_threshold"))
    bbx_loss = DetectionLoss(roi_config.getfloat("sigma"))
    msk_loss = InstanceSegLoss()
    lbl_roi_size = tuple(s * 2 for s in roi_size)
    roi_algo = InstanceSegAlgoFPN(bbx_prediction_generator, msk_prediction_generator, proposal_matcher, bbx_loss, msk_loss, classes,
                                  roi_config.getstruct("bbx_reg_weights"), roi_config.getint("fpn_canonical_scale"),
                                  roi_config.getint("fpn_canonical_level"), roi_size, roi_config.getint("fpn_min_level"),
                                  roi_config.getint("fpn_levels"), lbl_roi_size, roi_config.getboolean("void_is_background"), args.debug)
    roi_head = FPNMaskHead(transformer_config.getint("bev_ms_channels"), classes, roi_size, norm_act=norm_act_dynamic)

    # Create semantic segmentation network
    W_out = int(dl_config.getstruct("bev_crop")[0] * dl_config.getfloat("scale"))
    Z_out = int(dl_config.getstruct("bev_crop")[1] * dl_config.getfloat("scale"))
    out_shape = (W_out, Z_out)
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"), out_shape=out_shape, bev_params=bev_params,
                               extrinsics=extrinsics)
    sem_algo = SemanticSegAlgo(sem_loss, classes["total"])
    sem_head = FPNSemanticHeadDPC(transformer_config.getint("bev_ms_channels"),
                                  sem_config.getint("fpn_min_level"),
                                  sem_config.getint("fpn_levels"),
                                  classes["total"],
                                  out_size=out_shape,
                                  pooling_size=sem_config.getstruct("pooling_size"),
                                  norm_act=norm_act_static)

    # Panoptic fusion algorithm
    po_loss = PanopticLoss(classes["stuff"])
    po_fusion_algo = PanopticFusionAlgo(po_loss, classes["stuff"], classes["thing"], 1)

    # Create the BEV network
    return DvtNet(body, bev_transformer, rpn_head, roi_head, sem_head, transformer_algo, rpn_algo, roi_algo,
                          sem_algo, po_fusion_algo, args.train_dataset, classes=classes,
                          front_vertical_classes=transformer_config.getstruct("front_vertical_classes"),
                          front_flat_classes=transformer_config.getstruct("front_flat_classes"),
                          bev_vertical_classes=transformer_config.getstruct('bev_vertical_classes'),
                          bev_flat_classes=transformer_config.getstruct("bev_flat_classes"))


def make_optimizer(config, model, epoch_length):
    opt_config = config["optimizer"]
    sch_config = config["scheduler"]

    optimizer = optim.SGD(model.parameters(), lr=opt_config.getfloat("base_lr"),
                          weight_decay=opt_config.getfloat("weight_decay"))

    scheduler = scheduler_from_config(sch_config, optimizer, epoch_length)

    assert sch_config["update_mode"] in ("batch", "epoch")
    batch_update = sch_config["update_mode"] == "batch"
    total_epochs = sch_config.getint("epochs")

    return optimizer, scheduler, batch_update, total_epochs


def freeze_modules(args, model):
    for module in args.freeze_modules:
        print("Freezing module: {}".format(module))
        for name, param in model.named_parameters():
            if name.startswith(module):
                param.requires_grad = False

    # Freeze the dummy parameters
    for name, param in model.named_parameters():
        if name.endswith("dummy.weight"):
            param = torch.ones_like(param)
            param.requires_grad = False
            print("Freezing layer: {}".format(name))
        elif name.endswith("dummy.bias"):
            param = torch.zeros_like(param)
            param.requires_grad = False
            print("Freezing layer: {}".format(name))

    return model


def log_iter(mode, meters, time_meters, results, metrics, batch=True, **kwargs):
    assert mode in ['train', 'val', 'test'], "Mode must be either 'train', 'val', or 'test'!"
    iou = ["sem_conf"]

    log_entries = []

    if kwargs['lr'] is not None:
        log_entries = [("lr", kwargs['lr'])]

    meters_keys = list(meters.keys())
    meters_keys.sort()
    for meter_key in meters_keys:
        if meter_key in iou:
            log_key = meter_key
            log_value = meters[meter_key].iou.mean().item()
        else:
            if not batch:
                log_value = meters[meter_key].mean.item()
            else:
                log_value = meters[meter_key]
            log_key = meter_key

        log_entries.append((log_key, log_value))

    time_meters_keys = list(time_meters.keys())
    time_meters_keys.sort()
    for meter_key in time_meters_keys:
        log_key = meter_key
        if not batch:
            log_value = time_meters[meter_key].mean.item()
        else:
            log_value = time_meters[meter_key]
        log_entries.append((log_key, log_value))

    if metrics is not None:
        metrics_keys = list(metrics.keys())
        metrics_keys.sort()
        for metric_key in metrics_keys:
            log_key = metric_key
            if not batch:
                log_value = metrics[log_key].mean.item()
            else:
                log_value = metrics[log_key]
            log_entries.append((log_key, log_value))

    logging.iteration(kwargs["summary"], mode, kwargs["global_step"], kwargs["epoch"] + 1, kwargs["num_epochs"],
                      kwargs['curr_iter'], kwargs['num_iters'], OrderedDict(log_entries))


def train(model, optimizer, scheduler, dataloader, meters, **varargs):
    model.train()
    if not varargs['debug']:
        dataloader.batch_sampler.set_epoch(varargs["epoch"])
    optimizer.zero_grad()

    global_step = varargs["global_step"]
    loss_weights = varargs['loss_weights']

    time_meters = {"data_time": AverageMeter((), meters["loss"].momentum),
                   "batch_time": AverageMeter((), meters["loss"].momentum)}

    data_time = time.time()

    for it, sample in enumerate(dataloader):
        sample = {k: sample[k].cuda(device=varargs['device'], non_blocking=True) for k in NETWORK_INPUTS}
        sample['calib'], _ = pad_packed_images(sample['calib'])

        # Log the time
        time_meters['data_time'].update(torch.tensor(time.time() - data_time))

        # Update scheduler
        global_step += 1
        if varargs["batch_update"]:
            scheduler.step(global_step)

        batch_time = time.time()

        # Run network
        losses, results, stats = model(**sample, do_loss=True, do_prediction=False)
        if not varargs['debug']:
            distributed.barrier()

        losses = OrderedDict((k, v.mean() if v is not None else torch.tensor(0, device=varargs['device'])) for k, v in losses.items())
        losses["loss"] = sum(loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys())

        # Increment the optimiser and back propagate the gradients
        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        time_meters['batch_time'].update(torch.tensor(time.time() - batch_time))

        # Gather stats from all workers
        if not varargs['debug']:
            losses = all_reduce_losses(losses)

        sem_conf_stat = stats['sem_conf']

        # Gather the stats from all the workers
        if not varargs['debug']:
            distributed.all_reduce(sem_conf_stat, distributed.ReduceOp.SUM)

        # Update meters
        with torch.no_grad():
            for loss_name, loss_value in losses.items():
                meters[loss_name].update(loss_value.cpu())
            meters['sem_conf'].update(sem_conf_stat.cpu())

        # Clean-up
        del losses, stats, sample

        # Log to tensorboard and console
        if (it + 1) % varargs["log_interval"] == 0:
            if varargs["summary"] is not None:
                log_iter("train", meters, time_meters, results, None, batch=True, global_step=global_step,
                         epoch=varargs["epoch"], num_epochs=varargs['num_epochs'], lr=scheduler.get_lr()[0],
                         curr_iter=it+1, num_iters=len(dataloader), summary=varargs['summary'])

        data_time = time.time()

    del results
    return global_step


def validate(model, dataloader, **varargs):
    model.eval()

    if not varargs['debug']:
        dataloader.batch_sampler.set_epoch(varargs["epoch"])

    num_stuff = dataloader.dataset.num_stuff
    num_thing = dataloader.dataset.num_thing
    num_classes = num_stuff + num_thing

    loss_weights = varargs['loss_weights']

    val_meters = {
        "loss": AverageMeter(()),
        "obj_loss": AverageMeter(()),
        "bbx_loss": AverageMeter(()),
        "roi_cls_loss": AverageMeter(()),
        "roi_bbx_loss": AverageMeter(()),
        "roi_msk_loss": AverageMeter(()),
        "sem_loss": AverageMeter(()),
        "po_loss": AverageMeter(()),
        "sem_conf": ConfusionMatrixMeter(num_classes),
        "vf_loss": AverageMeter(()),
        "v_region_loss": AverageMeter(()),
        "f_region_loss": AverageMeter(())
    }

    time_meters = {"data_time": AverageMeter(()),
                   "batch_time": AverageMeter(())}

    # Validation metrics
    val_metrics = {"po_miou": AverageMeter(()), "sem_miou": AverageMeter(()),
                   "pq": AverageMeter(()), "pq_stuff": AverageMeter(()), "pq_thing": AverageMeter(()),
                   "sq": AverageMeter(()), "sq_stuff": AverageMeter(()), "sq_thing": AverageMeter(()),
                   "rq": AverageMeter(()), "rq_stuff": AverageMeter(()), "rq_thing": AverageMeter(())}

    # Accumulators for AP, mIoU and panoptic computation
    panoptic_buffer = torch.zeros(4, num_classes, dtype=torch.double)
    po_conf_mat = torch.zeros(256, 256, dtype=torch.double)
    sem_conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.double)

    data_time = time.time()

    for it, sample in enumerate(dataloader):
        batch_sizes = [m.shape[-2:] for m in sample['bev_msk']]
        original_sizes = sample['size']
        idxs = sample['idx']
        with torch.no_grad():
            sample = {k: sample[k].cuda(device=varargs['device'], non_blocking=True) for k in NETWORK_INPUTS}
            sample['calib'], _ = pad_packed_images(sample['calib'])

            time_meters['data_time'].update(torch.tensor(time.time() - data_time))
            batch_time = time.time()

            # Run network
            losses, results, stats = model(**sample, do_loss=True, do_prediction=True)

            if not varargs['debug']:
                distributed.barrier()

            losses = OrderedDict((k, v.mean() if v is not None else torch.tensor(0, device=varargs['device'])) for k, v in losses.items())
            losses["loss"] = sum(loss_weights[loss_name] * losses[loss_name] for loss_name in losses.keys())

            time_meters['batch_time'].update(torch.tensor(time.time() - batch_time))

            # Separate the normal and abnormal stats entries
            sem_conf_stat = stats['sem_conf']
            rem_stats = {k: v for k, v in stats.items() if k is not "sem_conf"}
            if not varargs['debug']:
                distributed.all_reduce(sem_conf_stat, distributed.ReduceOp.SUM)

            # Add the semantic confusion matrix to the existing one
            sem_conf_mat += sem_conf_stat.cpu()

            # Update meters
            with torch.no_grad():
                for loss_name, loss_value in losses.items():
                    val_meters[loss_name].update(loss_value.cpu())
                for stat_name, stat_value in rem_stats.items():
                    val_meters[stat_name].update(stat_value.cpu())
                val_meters['sem_conf'].update(sem_conf_stat.cpu())

            del losses, stats

            # Do the post-processing
            # panoptic_pred_list = panoptic_post_processing(results, idxs, sample['bev_msk'], sample['cat'],
            #                                               sample["iscrowd"])


            # Get the evaluation metrics
            # panoptic_buffer, po_conf_mat = compute_panoptic_test_metrics(panoptic_pred_list, panoptic_buffer,
            #                                                                    po_conf_mat, num_stuff=num_stuff,
            #                                                                    num_classes=num_classes,
            #                                                                    batch_sizes=batch_sizes,
            #                                                                    original_sizes=original_sizes)

            # Log batch to tensorboard and console
            if (it + 1) % varargs["log_interval"] == 0:
                if varargs['summary'] is not None:
                    log_iter("val", val_meters, time_meters, results, None, global_step=varargs['global_step'],
                             epoch=varargs['epoch'], num_epochs=varargs['num_epochs'], lr=None, curr_iter=it+1,
                             num_iters=len(dataloader), summary=None)

            data_time = time.time()

    # Finalise Panoptic mIoU computation
    # po_conf_mat = po_conf_mat.to(device=varargs["device"])
    # if not varargs['debug']:
    #     distributed.all_reduce(po_conf_mat, distributed.ReduceOp.SUM)
    # po_conf_mat = po_conf_mat.cpu()[:num_classes, :]
    # po_intersection = po_conf_mat.diag()
    # po_union = ((po_conf_mat.sum(dim=1) + po_conf_mat.sum(dim=0)[:num_classes] - po_conf_mat.diag()) + 1e-8)
    # po_miou = po_intersection / po_union

    # Finalise semantic mIoU computation
    sem_conf_mat = sem_conf_mat.to(device=varargs['device'])
    if not varargs['debug']:
        distributed.all_reduce(sem_conf_mat, distributed.ReduceOp.SUM)
    sem_conf_mat = sem_conf_mat.cpu()[:num_classes, :]
    sem_intersection = sem_conf_mat.diag()
    sem_union = ((sem_conf_mat.sum(dim=1) + sem_conf_mat.sum(dim=0)[:num_classes] - sem_conf_mat.diag()) + 1e-8)
    sem_miou = sem_intersection / sem_union

    # Save the metrics
    scores = {}
    # scores['po_miou'] = po_miou.mean()
    scores['sem_miou'] = sem_miou.mean()
    scores = get_scores(panoptic_buffer, scores, varargs["device"], num_stuff, varargs['debug'])
    # Update the validation metrics meters
    for key in val_metrics.keys():
        if key in scores.keys():
            if scores[key] is not None:
                val_metrics[key].update(scores[key].cpu())

    # Log results
    log_info("Validation done", debug=varargs['debug'])
    if varargs["summary"] is not None:
        log_iter("val", val_meters, time_meters, None, val_metrics, batch=False, summary=varargs['summary'],
                 global_step=varargs['global_step'], curr_iter=len(dataloader), num_iters=len(dataloader),
                 epoch=varargs['epoch'], num_epochs=varargs['num_epochs'], lr=None)

    log_miou("Semantic mIoU", sem_miou, dataloader.dataset.categories)
    log_scores("Scores", scores)

    # return scores['pq'].item()
    return scores['sem_miou'].item()


def main(args):
    if not args.debug:
        # Initialize multi-processing
        distributed.init_process_group(backend='nccl', init_method='env://')
        device_id, device = args.local_rank, torch.device(args.local_rank)
        rank, world_size = distributed.get_rank(), distributed.get_world_size()
        torch.cuda.set_device(device_id)
    else:
        rank = 0
        world_size = 1
        device_id, device = rank, torch.device(rank+3)

    # Create directories
    if not args.debug:
        log_dir, saved_models_dir, config_file = create_run_directories(args, rank)
    else:
        config_file = os.path.join(args.project_root_dir, "experiments", "config", args.config)

    # Load configuration
    config = make_config(args, config_file)

    # Initialize logging only for rank 0
    if not args.debug and rank == 0:
        logging.init(log_dir, "train" if args.mode == 'train' else "test")
        summary = tensorboard.SummaryWriter(log_dir)
    else:
        summary = None

    # Create dataloaders
    train_dataloader, val_dataloader = make_dataloader(args, config, rank, world_size)

    # Create model
    model = make_model(args, config, train_dataloader.dataset.num_thing, train_dataloader.dataset.num_stuff)

    # Freeze modules based on the argument inputs
    model = freeze_modules(args, model)

    if args.resume:
        assert not args.pre_train, "resume and pre_train are mutually exclusive"
        log_info("Loading snapshot from %s", args.resume, debug=args.debug)
        snapshot = resume_from_snapshot(model, args.resume, ["body", "transformer", "rpn_head", "roi_head", "sem_head"])
    elif args.pre_train:
        assert not args.resume, "resume and pre_train are mutually exclusive"
        log_info("Loading pre-trained model from %s", args.pre_train, debug=args.debug)
        pre_train_from_snapshots(model, args.pre_train, ["body", "transformer", "rpn_head", "roi_head", "sem_head"], rank)
    else:
        assert not args.eval, "--resume is needed in eval mode"
        snapshot = None

    # Init GPU stuff
    if not args.debug:
        torch.backends.cudnn.benchmark = config["general"].getboolean("cudnn_benchmark")
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # Convert batch norm to SyncBatchNorm
        model = DistributedDataParallel(model.cuda(device), device_ids=[device_id], output_device=device_id,
                                        find_unused_parameters=True)
    else:
        model = model.cuda(device)

    # Create optimizer
    optimizer, scheduler, batch_update, total_epochs = make_optimizer(config, model, len(train_dataloader))
    if args.resume:
        optimizer.load_state_dict(snapshot["state_dict"]["optimizer"])

    # Training loop
    momentum = 1. - 1. / len(train_dataloader)
    train_meters = {
        "loss": AverageMeter((), momentum),
        "obj_loss": AverageMeter((), momentum),
        "bbx_loss": AverageMeter((), momentum),
        "roi_cls_loss": AverageMeter((), momentum),
        "roi_bbx_loss": AverageMeter((), momentum),
        "roi_msk_loss": AverageMeter((), momentum),
        "sem_loss": AverageMeter((), momentum),
        "po_loss": AverageMeter((), momentum),
        "sem_conf": ConfusionMatrixMeter(train_dataloader.dataset.num_categories, momentum),
        "vf_loss": AverageMeter((), momentum),
        "v_region_loss": AverageMeter((), momentum),
        "f_region_loss": AverageMeter((), momentum)
    }

    if args.resume:
        starting_epoch = snapshot["training_meta"]["epoch"] + 1
        best_score = snapshot["training_meta"]["best_score"]
        global_step = snapshot["training_meta"]["global_step"]
        for name, meter in train_meters.items():
            meter.load_state_dict(snapshot["state_dict"][name + "_meter"])
        del snapshot
    else:
        starting_epoch = 0
        best_score = 0
        global_step = 0

    for epoch in range(starting_epoch, total_epochs):
        log_info("Starting epoch %d", epoch + 1, debug=args.debug)
        if not batch_update:
            scheduler.step(epoch)

        # Run training epoch
        global_step = train(model, optimizer, scheduler, train_dataloader, train_meters,
                            batch_update=batch_update, epoch=epoch, summary=summary, device=device,
                            log_interval=config["general"].getint("log_interval"), num_epochs=total_epochs,
                            global_step=global_step, loss_weights=config['optimizer'].getstruct("loss_weights"),
                            log_train_samples=config['general'].getboolean("log_train_samples"),
                            front_vertical_classes=config['transformer'].getstruct('front_vertical_classes'),
                            front_flat_classes=config['transformer'].getstruct('front_flat_classes'),
                            bev_vertical_classes=config['transformer'].getstruct('bev_vertical_classes'),
                            bev_flat_classes=config['transformer'].getstruct('bev_flat_classes'),
                            rgb_mean=config['dataloader'].getstruct('rgb_mean'),
                            rgb_std=config['dataloader'].getstruct('rgb_std'),
                            img_scale=config['dataloader'].getfloat('scale'),
                            debug=args.debug)

        # Save snapshot (only on rank 0)
        if not args.debug and rank == 0:
            snapshot_file = path.join(saved_models_dir, "model_latest.pth")
            log_info("Saving snapshot to %s", snapshot_file)
            meters_out_dict = {k + "_meter": v.state_dict() for k, v in train_meters.items()}
            save_snapshot(snapshot_file, config, epoch, 0, best_score, global_step,
                          body=model.module.body.state_dict(),
                          transformer=model.module.transformer.state_dict(),
                          rpn_head=model.module.rpn_head.state_dict(),
                          roi_head=model.module.roi_head.state_dict(),
                          sem_head=model.module.sem_head.state_dict(),
                          optimizer=optimizer.state_dict(),
                          **meters_out_dict)

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            saved_models_dir = None if args.debug else saved_models_dir
            log_info("Validating epoch %d", epoch + 1, debug=args.debug)
            score = validate(model, val_dataloader, device=device, summary=summary, global_step=global_step,
                             epoch=epoch, num_epochs=total_epochs,log_interval=config["general"].getint("log_interval"),
                             loss_weights=config['optimizer'].getstruct("loss_weights"),
                             front_vertical_classes=config['transformer'].getstruct('front_vertical_classes'),
                             front_flat_classes=config['transformer'].getstruct('front_flat_classes'),
                             bev_vertical_classes=config['transformer'].getstruct('bev_vertical_classes'),
                             bev_flat_classes=config['transformer'].getstruct('bev_flat_classes'),
                             rgb_mean=config['dataloader'].getstruct('rgb_mean'),
                             rgb_std=config['dataloader'].getstruct('rgb_std'),
                             img_scale=config['dataloader'].getfloat('scale'),
                             debug=args.debug)

            # Update the score on the last saved snapshot
            if not args.debug and rank == 0:
                snapshot = torch.load(snapshot_file, map_location="cpu")
                snapshot["training_meta"]["last_score"] = score
                torch.save(snapshot, snapshot_file)
                del snapshot

                if score > best_score:
                    best_score = score
                    if rank == 0:
                        shutil.copy(snapshot_file, path.join(saved_models_dir, "model_best.pth"))

        if (epoch + 1) % config["general"].getint("val_interval") == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main(parser.parse_args())
