import os
import json
import time
import argparse
import torch
import shutil
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torchvision import transforms
from src.dataset import *
from src.model import CSRNet
from src.utils import *

parser = argparse.ArgumentParser(description = "PyTorch CSRNet vehicle")
parser.add_argument("--train_json", 
            default = "/data/wangyf/datasets/TRANCOS_v3/image_sets/training.txt",
                    help = "path to train json")
parser.add_argument("--val_json", 
        default = "/data/wangyf/datasets/TRANCOS_v3/image_sets/validation.txt", 
                    help = "path to val json")
parser.add_argument("--gpu", default = '6', type = str, help = "GPU id to use.")
parser.add_argument("--task", default = 's_1_100_0320', type = str, 
                    help = "task id to use.")
parser.add_argument("--pre", '-p', default = None, type = str, 
                    help = "path to pretrained model")
parser.add_argument("--lr", default = 1e-6, type = float, help = "original learning rate")                 ## TODO
parser.add_argument("--epochs_drop", default = 30, type = int, help = "epochs_drop")                       ## TODO
parser.add_argument("--start_epoch", default = 0, type = int, help = "start epoch")
parser.add_argument("--epochs", default = 60,type = int, help = "epoch")                                  ## TODO
parser.add_argument("--batch_size", default = 1, type = int, help = "batch size")                          ## TODO
parser.add_argument("--momentum", default = 0.95, type = float)                                            ## TODO
parser.add_argument("--decay", default = 5 * 1e-4, type = float)                                           ## TODO
parser.add_argument("--workers", default = 4,type = int)                                                  ## TODO
parser.add_argument("--print_freq", default = 40, type = int, help = "show train log information")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


try:
    from termcolor import cprint
except ImportError:
    cprint = None

## create log for ssh check
localtime = time.strftime("%Y-%m-%d", time.localtime())
log_file = open("./logs/record" + localtime + ".txt", 'w')

## create tensorboard dir
logDir = "./tblogs/0320"
if os.path.exists(logDir):
    shutil.rmtree(logDir)  # remove recursive dir
writer = SummaryWriter(logDir)

def log_print(text, color = None, on_color = None,
              attrs = None, log_file = log_file):
    print(text, file = log_file)
    if cprint is not None:
        cprint(text, color = color, on_color = on_color, attrs = attrs)
    else:
        print(text)

# .txt(str) convert to list, list is the image path
train_list = [] # 403
val_list = []   # 420
with open(args.train_json, 'r') as f1:
    for line in f1.readlines():
        line = line.strip("\n")
        train_list.append(line)
with open(args.val_json, 'r') as f2:
    for line in f2.readlines():
        line = line.strip("\n")
        val_list.append(line)


# Sets the learning rate to the initial LR decayed by 10 every 30 epochs
def adjust_learning_rate(optimizer, epoch):
    factor = 0.1 ** (epoch // args.epochs_drop)
    args.lr = args.lr * factor
    # print(len(optimizer.param_groups))  1
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

###  Computes and stores the average and current value
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.cur = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, cur, n = 1):
        self.cur = cur
        self.sum += cur * n
        self.count += n
        self.avg = self.sum / self.count

###  training
def train(train_list, model, criterion, optimizer, epoch):
    '''
    losses: batch_size loss value, include cur and avg
    batch_time: batch_size train time, include cur and avg
    data_time: batch_size loader input time, include cur and avg
    '''
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean, std)
                                   ])
    DatasetLoader_train = listDataset(train_list,
                                      shuffle = True,
                                      transform = transform_train,
                                      train = True,
                                      seen = model.seen,
                                      batch_size = args.batch_size,
                                      num_workers = args.workers)
    train_loader = torch.utils.data.DataLoader(DatasetLoader_train,
                                               batch_size = args.batch_size)

    log_text = "epoch %d, processed %d samples, lr % .10f "\
               %(epoch, epoch * len(train_loader.dataset), args.lr)
    log_print(log_text, color = "green", attrs = ["bold"])
    model.train()
    end = time.time()
    for i,(img, target)in enumerate(train_loader):
        '''img: ; target: torch.Size([1, H/8, W/8])'''
        data_time.update(time.time()- end)
        img = img.cuda()                                                # torch.Size([batch_size,3,H,W])
        img = Variable(img)                                             # torch.FloatTensor to Variable
        output = model(img)                                             # torch.Size([batch_size,1,H/8,W/8])
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()     # torch.Size([1,batch_size,H/8,W/8])
        target = Variable(target)
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))                         # img.size[0] == batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_str = "Epoch: [{0}][{1}/{2}]\t" \
                .format(epoch, i, len(train_loader))
            print_str += "Data time {data_time.cur:.3f}({data_time.avg:.3f})\t" \
                .format(data_time=data_time)
            print_str += "Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t" \
                .format(batch_time=batch_time)
            print_str += "Loss {loss.cur:.4f}({loss.avg:.4f})\t" \
                .format(loss=losses)
            # print(print_str)
            log_print(print_str, color="green", attrs=["bold"])

    return losses.avg

###  val
def validate(val_list, model, epoch, criterion):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    DatasetLoader_val = listDataset(val_list,
                                    shuffle = False,
                                    transform = transform_val,
                                    train = False)
    val_loader = torch.utils.data.DataLoader(DatasetLoader_val,
                                             batch_size = args.batch_size)                       # [C,H,W]
    model.eval()
    end = time.time()
    mae = 0.0
    for i,(img, gt_density_map) in enumerate(val_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        gt_density_map = gt_density_map.type(torch.FloatTensor).unsqueeze(1)
        gt_density_map = gt_density_map.cuda()
        gt_density_map = Variable(gt_density_map)
        et_density_map = model(img)
        loss = criterion(gt_density_map, et_density_map)
        losses.update(loss.item(), img.size(0))
        batch_time.update(time.time() - end)
        mae += abs(et_density_map.data.sum() - gt_density_map.sum())  # todo
        end = time.time()
        if i % args.print_freq == 0:
            print_str = "Epoch: [{0}][{1}/{2}]\t" \
                .format(epoch, i, len(val_loader))
            print_str += "Data time {data_time.cur:.3f}({data_time.avg:.3f})\t" \
                .format(data_time=data_time)
            print_str += "Batch time {batch_time.cur:.3f}({batch_time.avg:.3f})\t" \
                .format(batch_time=batch_time)
            print_str += "Loss {loss.cur:.4f}({loss.avg:.4f})\t" \
                .format(loss=losses)
            # print(print_str)
            log_print(print_str, color="red", attrs=["bold"])
    mae = mae / len(val_loader)
    return losses.avg, mae


def main():
    global best_mae
    best_mae = 1e6
    seed = time.time()
    torch.cuda.manual_seed(seed)
    model = CSRNet()                                                    # CSRNet[weight, bias] Initialize
    model = model.cuda()
    criterion = SANetLoss(1).cuda()
    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum = args.momentum,
                                weight_decay = args.decay)  ## TODO

    ##  pre-train model
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_mae = checkpoint['best_mae']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})" .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_list, model, criterion, optimizer, epoch)
        val_loss, val_mae = validate(val_list, model, epoch, criterion)
        writer.add_scalar("/train_loss", train_loss, epoch)
        writer.add_scalar("/val_loss", val_loss, epoch)
        is_best = val_mae < best_mae
        best_mae = min(val_mae, best_mae)
        print(epoch, val_mae, best_mae)
        ## save model 1-400
        save_checkpoint({"epoch": epoch + 1,
                         "arch": args.pre,
                         "state_dict": model.state_dict(),
                         "best_mae": best_mae,
                         "optimizer": optimizer.state_dict(), },
                        is_best,
                        args.task)
if __name__ == '__main__':
    main()        
