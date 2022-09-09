import argparse
import os.path
import torch
from mmcv.utils import Config
from mmseg.datasets import build_dataset, build_dataloader
from loguru import logger
import time
import datetime
from PIL import Image
import numpy as np
from tqdm import tqdm

import models
import datasets
import utils


def train(config):
    trace = logger.add(os.path.join(config.work_dir, 'logs.log'))
    logger.info(f'Building datasets.')
    dataset_ga = build_dataset(config.data.train_ga)
    if config.data.train_normal_expand.data_root is not None:
        dataset_normal = build_dataset([config.data.train_normal, config.data.train_normal_expand])
    else:
        dataset_normal = build_dataset(config.data.train_normal)
    logger.info(f'{dataset_ga.__len__():d} GA images for training.')
    logger.info(f'{dataset_normal.__len__():d} normal images for training.')

    loader_train_ga = build_dataloader(
        dataset_ga,
        samples_per_gpu=config.batch_size,
        workers_per_gpu=config.num_workers,
        shuffle=True,
        drop_last=True)
    loader_train_normal = build_dataloader(
        dataset_normal,
        samples_per_gpu=config.batch_size,
        workers_per_gpu=config.num_workers,
        shuffle=True,
        drop_last=True)

    logger.info(f"Creating model: {config.model.type}")
    model = models.build_model(config.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    model.cuda()

    logger.info(f"Creating optimizer: {config.optimizer.type}")
    optimizer = utils.build_optimizer(config.optimizer, model)

    criterion_class = torch.nn.CrossEntropyLoss()
    criterion_decoder = torch.nn.MSELoss()
    criterion_neck = models.DictionaryLoss(use_entropy=False)

    save_freq = 1
    log_freq = 10
    logger.info(f"Start training for {config.epochs:d} epochs.")
    num_steps = min(dataset_ga.__len__(), dataset_normal.__len__()) // config.batch_size
    for epoch in range(config.epochs):
        logger.info(f'Fold [{config.fold:d}] Epoch [{epoch:d}]')
        start = time.time()
        end = time.time()
        model.train()
        for i, data in enumerate(zip(loader_train_ga, loader_train_normal)):
            img_ga = data[0]['img'].to('cuda', non_blocking=True)
            img_normal = data[1]['img'].to('cuda', non_blocking=True)
            class_ga = torch.tensor([0., 1.]).expand((config.batch_size, 2)).to('cuda', non_blocking=True)
            class_normal = torch.tensor([1., 0.]).expand((config.batch_size, 2)).to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                model.forward_train(anchor=img_normal, contrast=img_ga)
                loss_class_anchor = criterion_class(model.out_class_anchor, class_normal)
                loss_class_contrast = criterion_class(model.out_class_contrast, class_ga)
                loss_neck_anchor = criterion_neck(model.apm_anchor, is_positive=False)
                loss_neck_contrast = criterion_neck(model.apm_contrast, is_positive=True)
                loss_decoder_anchor = criterion_decoder(model.out_decoder_anchor, img_normal)
                loss_decoder_contrast = criterion_decoder(model.out_decoder_contrast, img_ga)

                loss_class = loss_class_anchor + loss_class_contrast
                loss_neck = loss_neck_anchor + loss_neck_contrast
                loss_decoder = loss_decoder_anchor + loss_decoder_contrast
                loss = loss_class + loss_decoder + loss_neck

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % log_freq == 0:
                logger.info(
                    f'Train: [{epoch}/{config.epochs}][{i}/{num_steps}] '
                    f'eta {datetime.timedelta(seconds=int(batch_time * (num_steps - i)))} '
                    f'time {batch_time:.4f} '
                    f'loss [cls {loss_class.item():.4f} neck {loss_neck.item():.4f} decoder {loss_decoder.item():.4f}] '
                    f'mem {memory_used:.0f}MB')
        epoch_time = time.time() - start
        logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

        if epoch % save_freq == 0:
            checkpoint_paths = [os.path.join(config.work_dir, '{}.pth'.format(epoch)),
                                os.path.join(config.work_dir, 'checkpoint.pth')]
            for checkpoint in checkpoint_paths:
                utils.save_on_master({
                        'model': model.state_dict(),
                        'epoch': epoch,
                    }, checkpoint)

        time.sleep(1)

    logger.remove(trace)


@torch.no_grad()
def evaluate(data_loader, model, show_dir=None, num_frames=128):
    if show_dir is None:
        logger.info(f"show_dir is None")
        return

    # switch to evaluation mode
    model.eval()
    num_datas = data_loader.dataset.__len__()

    outputs = {}
    save_shape = {}
    for data in tqdm(data_loader):
        img = data['img'][0].to('cuda', non_blocking=True)

        with torch.cuda.amp.autocast():
            apm = model(img).squeeze(0).cpu().numpy()

        img_metas = data['img_metas'][0].data[0]
        for img_meta in img_metas:
            filename = img_meta['ori_filename'].split('/')[0]
            frame = img_meta['ori_filename'].split('/')[1].replace('.bmp', '')
            img_shape = img_meta['img_shape']
            ori_shape = img_meta['ori_shape']

            if filename not in outputs:
                outputs[filename] = np.zeros((img_shape[0], img_shape[1], num_frames))
                save_shape[filename] = (ori_shape[1], ori_shape[1])
            outputs[filename][:, :, int(frame) - 1] = apm

    for key in outputs:
        proj = np.sum(outputs[key], axis=0)
        proj = utils.normalize(proj)
        save_path = os.path.join(show_dir, key + '.bmp')
        Image.fromarray(proj).resize(save_shape[key], resample=Image.NEAREST).save(save_path)

    logger.info(f"evaluated {num_datas} Frames of {num_datas // num_frames} Cubes")


@torch.no_grad()
def test(config):
    assert config.show_dir is not None

    dataset_test = build_dataset(config.data.test)
    test_loader = build_dataloader(
        dataset_test,
        samples_per_gpu=1,
        workers_per_gpu=config.num_workers,
        shuffle=False)

    model = models.build_model(config.model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    model.cuda()

    load_path = os.path.join(config.work_dir, 'checkpoint.pth')
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    evaluate(test_loader, model, show_dir=config.show_dir, num_frames=config.num_frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GA Segmentation training and evaluation script')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--show-dir', type=str)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    if args.show_dir is not None:
        cfg.show_dir = args.show_dir
    else:
        cfg.show_dir = None
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers

    train(cfg)
    test(cfg)
