import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import numpy as np
import torchvision.models.segmentation
import torch
import matplotlib.pyplot as plt
import argparse
from torch.nn import DataParallel as DP

# PARAMETERS
# ---------------------------------------------------------------------------
Learning_Rate = 1e-5
batchSize = 8
height = 1280//2
width = 1920//2
images_dir = "../opt_dataset/raw_images"
# ---------------------------------------------------------------------------


def getLatestCkptNum():
    all_ckpts_names = os.listdir(ckpts_dir)
    if len(all_ckpts_names) == 0:
        return 0
    lll = [int(bla[0:-3]) for bla in all_ckpts_names]
    lll.sort()
    return lll[-1]


all_image_names = os.listdir(images_dir)
m = len(all_image_names)

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--action", help="Action? train/trainplus", default="train")
parser.add_argument("-c", "--cuda", help="CUDA Number", required=True)
parser.add_argument("-e", "--epochs", help="The final total epochs you want", required=True, type=int)
parser.add_argument("-n", "--num", help="Saved Ckpt Number", type=int, default=getLatestCkptNum())
parser.add_argument("-b", "--batch", help="Batch Size", type=int, default=batchSize)
parser.add_argument("-m", "--model", help="Model name", required=True)

args = parser.parse_args()

bmask_dir = f"../opt_dataset/bmasks/{args.model}"
ckpts_dir = f"../ckpts/{args.model}"


def GetInput(idx, device):
    img = torch.load(os.path.join(images_dir, all_image_names[idx]), map_location=device)
    bmask = torch.load(os.path.join(bmask_dir, all_image_names[idx]), map_location=device)
    return img, bmask


def LoadBatch(indices, device):
    bsize = len(indices)
    images = torch.zeros([bsize, 3, height, width])
    ann = torch.zeros([bsize, height, width])
    for i in range(bsize):
        images[i], ann[i] = GetInput(indices[i], device)
    return images, ann


def save_ckpt(state, epoch):
    cpath = ckpts_dir + "/" + str(epoch) + ".pt"
    torch.save(state, cpath)


def load_ckpt(cpath, model, optimizer):
    ckpt = torch.load(cpath)
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])
    ini_epoch = ckpt['epoch']
    return model, optimizer, ini_epoch


def _train(Net, optimizer, ini_epoch, device, append=False):
    history_loss = []
    GLOBAL_loss = []
    print(f"Would be doing {args.epochs - args.num} epochs")
    for epoch in range(ini_epoch, args.epochs):
        permutation = torch.randperm(m)
        for i in range(0, m, batchSize):
            start = i
            if i+batchSize-1 > m-1:
                stop = m-1
            else:
                stop = i+batchSize-1
            indices = permutation[start:stop+1]
            images, ann = LoadBatch(indices, device)  # Load taining batch
            images = torch.autograd.Variable(
                images, requires_grad=False).to(device)  # Load image
            ann = torch.autograd.Variable(
                ann, requires_grad=False).to(device)  # Load annotation
            Pred = Net(images)['out']  # make prediction
            Net.zero_grad()
            criterion = torch.nn.CrossEntropyLoss()  # Set loss function
            Loss = criterion(Pred, ann.long())  # Calculate cross entropy loss
            history_loss.append(Loss)
            GLOBAL_loss.append(Loss)
            Loss.backward()  # Backpropogate loss
            optimizer.step()  # Apply gradient descent change to weight
            print("Epoch: " + str(epoch) + "  Iter: " +
                  str(i//batchSize) + "  Loss=", Loss.data.cpu().numpy())
        state = {'epoch': epoch+1, 'state_dict': Net.state_dict(),
                 'optimizer': optimizer.state_dict(), 'losses': history_loss}
        if epoch % 500 == 0:
            save_ckpt(state, epoch)
            history_loss = []
    plt.plot(GLOBAL_loss)
    plt.savefig(f"{ckpts_dir}/loss.png")
    plt.semilogx(GLOBAL_loss)
    plt.savefig(f"{ckpts_dir}/loss_semilogx.png")
    plt.semilogy(GLOBAL_loss)
    plt.savefig(f"{ckpts_dir}/loss_semilogy.png")


def _trainplus(cpath, Net, optimizer, device):
    Net, optimizer, ini_epoch = load_ckpt(cpath, Net, optimizer)
    _train(Net, optimizer, ini_epoch, device, append=True)


def getGPUids(gpus):
    ll = [int(s) for s in gpus.split(',') if s.isdigit()]
    print(f"Using GPUS {args.cuda} -> {ll}")
    return ll


ll = getGPUids(args.cuda)

device = torch.device(f'cuda:{ll[0]}') if torch.cuda.is_available() else torch.device('cpu')
Net = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  # Load net
Net.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))  # Change final layer to 2 classes
Net.backbone.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
Net = DP(Net, device_ids=ll)
Net = Net.to(device)
optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate)  # Create adam optimizer

if args.action == "train":
    _train(Net, optimizer, 0, device, append=False)
elif args.action == "trainplus":
    _trainplus(f"{ckpts_dir}/{args.num}.pt", Net, optimizer, device)
else:
    raise ValueError("Invalid argument for action!")
