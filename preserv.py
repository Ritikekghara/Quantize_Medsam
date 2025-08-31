import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from layers import *
from attention import *


from config import Config


parser = argparse.ArgumentParser(description='Tiny-ViT')
parser.add_argument('data', nargs='?', metavar='DIR', default='gts', help='path to dataset')
parser.add_argument('--quant', default=False, action='store_true')
parser.add_argument('--ptf', default=False, action='store_true')
parser.add_argument('--lis', default=False, action='store_true')
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--calib-batchsize',
                    default=2,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument('--calib-iter', default=2, type=int)
parser.add_argument('--val-batchsize',
                    default=1,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=4,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--model', default='tiny_vit', choices=['tiny_vit'], help='model name')
parser.add_argument('--pretrained-path', default='lite_medsam.pth', type=str,
                    help='path to a local pretrained .pth checkpoint (optional)')


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def str2model(name):
    d = {
        'tiny_vit': QuantImageEncoder,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


# Small NPZ dataset support: many medical datasets use .npz files with arrays.
class NPZDataset(torch.utils.data.Dataset):
    def __init__(self, folder, img_size=256, transform=None):
        self.files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npz')])
        self.img_size = img_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def _load_array(self, path):
        try:
            d = np.load(path)
        except Exception:
            return None
        # pick the first array inside the npz
        if isinstance(d, np.lib.npyio.NpzFile):
            keys = list(d.files)
            if len(keys) == 0:
                return None
            arr = d[keys[0]]
        else:
            arr = d

        arr = np.asarray(arr)
        # if grayscale, make 3-channel
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr, arr, arr], axis=2)

        # convert to uint8 for PIL
        if arr.dtype != np.uint8:
            mx = arr.max() if arr.size > 0 else 1.0
            if mx <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        return arr

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = self._load_array(path)
        if arr is None:
            # return a random tensor if load failed
            img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        else:
            img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        # dummy label 0 (no labels available)
        return img, 0


def make_loader(path, batch_size, shuffle=False, num_workers=4, pin_memory=True):
    # prefer NPZ files if present
    if os.path.isdir(path):
        npz_files = [f for f in os.listdir(path) if f.endswith('.npz')]
        if len(npz_files) > 0:
            ds = NPZDataset(path, img_size=256)
            return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    # fallback to ImageFolder
    ds = datasets.ImageFolder(path)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)


def has_class_folders_with_files(path):
    """Return True if `path` contains class subfolders with at least one file each.

    This helper is at module scope so any Dataset created from it is picklable by
    multiprocessing on Windows.
    """
    if not os.path.isdir(path):
        return False
    for entry in os.scandir(path):
        if entry.is_dir():
            # check if this class folder has at least one file
            try:
                if any(os.scandir(entry.path)):
                    return True
            except PermissionError:
                continue
    return False


def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)

    # If a local pretrained path is provided, build the model without
    # triggering the remote download and load the checkpoint safely.
    if args.pretrained_path:
        model = str2model(args.model)(pretrained=False, cfg=cfg)
        # safe load: prefer weights_only when available
        import inspect
        from collections import OrderedDict

        load_kwargs = {'map_location': 'cpu'}
        try:
            if 'weights_only' in inspect.signature(torch.load).parameters:
                load_kwargs['weights_only'] = True
        except Exception:
            pass

        checkpoint = torch.load(args.pretrained_path, **load_kwargs)
        if isinstance(checkpoint, dict):
            print('Loaded checkpoint keys:', list(checkpoint.keys())[:10])

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state = OrderedDict()
        for k, v in state_dict.items():
            # strip DataParallel prefix
            new_key = k[7:] if k.startswith('module.') else k
            new_state[new_key] = v

        # If checkpoint keys are namespaced (e.g. 'image_encoder.*'), strip that
        # namespace if the model's state_dict keys don't contain it.
        sample_key = next(iter(new_state.keys())) if len(new_state) > 0 else ''
        if sample_key.startswith('image_encoder.'):
            stripped = OrderedDict()
            for k, v in new_state.items():
                stripped_key = k[len('image_encoder.'):] if k.startswith('image_encoder.') else k
                stripped[stripped_key] = v
            new_state = stripped

        # Filter checkpoint to only the keys that exist in the model to avoid
        # unexpected names (e.g., mask_decoder.*) causing noise.
        model_keys = set(model.state_dict().keys())
        filtered_state = OrderedDict()
        kept = []
        dropped = []
        for k, v in new_state.items():
            if k in model_keys:
                filtered_state[k] = v
                kept.append(k)
            else:
                dropped.append(k)

        if kept:
            print(f'Kept {len(kept)} keys from checkpoint (example):', kept[:10])
        if dropped:
            print(f'Dropped {len(dropped)} keys not present in model (example):', dropped[:10])

        load_res = model.load_state_dict(filtered_state, strict=False)
        try:
            missing = getattr(load_res, 'missing_keys', None)
            unexpected = getattr(load_res, 'unexpected_keys', None)
            if missing:
                print('Missing keys when loading checkpoint:', missing[:10])
            if unexpected:
                print('Unexpected keys in checkpoint:', unexpected[:10])
        except Exception:
            # older torch may return dict; pretty-print if so
            print('Load result:', load_res)
    else:
        model = str2model(args.model)(pretrained=True, cfg=cfg)

    model = model.to(device)


    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # (uses module-level `NPZDataset`, `make_loader`, and `has_class_folders_with_files`)
    # If valdir contains ImageFolder-style class subfolders use ImageFolder.
    # If it contains .npz files use NPZDataset.
    if not has_class_folders_with_files(valdir):
        # check for npz files
        if os.path.isdir(valdir) and any(f.endswith('.npz') for f in os.listdir(valdir)):
            print(f"Validation directory '{valdir}' contains .npz files; using NPZDataset for validation.")
            val_loader = make_loader(valdir, batch_size=args.val_batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        else:
            print(f"Validation directory '{valdir}' not found or doesn't contain class subfolders with images. Running a dry run on random input and exiting.")
            def dry_run(model, device, batch_size, img_size=256):
                model.eval()
                img_size = getattr(model, 'img_size', img_size)
                x = torch.randn(batch_size, 3, img_size, img_size, device=device)
                with torch.no_grad():
                    out = model(x)
                print(f"Dry run completed. Output shape: {tuple(out.shape)}")

            dry_run(model, device, batch_size=args.val_batchsize)
            return
    else:
        val_loader = make_loader(valdir, batch_size=args.val_batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)


    # switch to evaluate mode
    model.eval()
    

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)


    if args.quant:
        # Use NPZ-aware loader if traindir contains .npz, otherwise ImageFolder
        train_loader = make_loader(traindir, batch_size=args.calib_batchsize, shuffle=True, num_workers=args.num_workers, pin_memory=True)
        # ensure drop_last for calibration
        # If make_loader returned an ImageFolder DataLoader we can't easily set drop_last here,
        # but calibration loop tolerates it; otherwise wrap into a DataLoader with drop_last.
        if not getattr(train_loader, 'drop_last', False):
            # Create a new DataLoader with same dataset and desired drop_last
            train_loader = torch.utils.data.DataLoader(
                train_loader.dataset,
                batch_size=args.calib_batchsize,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        # Get calibration set.
        image_list = []
        for i, (data, target) in enumerate(train_loader):
            if i == args.calib_iter:
                break
            data = data.to(device)
            image_list.append(data)

        print('Calibrating...')
        model.model_open_calibrate()
        with torch.no_grad():
            for i, image in enumerate(image_list):
                if i == len(image_list) - 1:
                    # This is used for OMSE method to
                    # calculate minimum quantization error
                    model.model_open_last_calibrate()
                output = model(image)
        model.model_close_calibrate()
        model.model_quant()

    print('Validating...')
    val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device)
    # Print a concise summary so the script always outputs final results
    print(f"Validation finished. Loss: {val_loss:.4f}, Prec@1: {val_prec1:.3f}, Prec@5: {val_prec5:.3f}")
    # print("model",model)
    


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        # If model returns a spatial map (B, C, H, W), convert to (B, C)
        # by global average pooling so classification losses work with
        # a scalar target per image.
        if output.dim() == 4:
            output = F.adaptive_avg_pool2d(output, 1).view(output.size(0), -1)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




if __name__ == '__main__':
    main()


