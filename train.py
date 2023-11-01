





import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from dataset import *
import time
# from model import *
from models import SEW
from utlis import seed_all
import os
os.system('wandb login f1ff739b893fd48fb835c7cb39cbe54968b34c44')



# Argument parsing
parser = argparse.ArgumentParser(description='SNN training')
# 数据
parser.add_argument('--data_path', type=str, default='./dataset', help='learning rate')
parser.add_argument('--data_name', type=str, default='cifar10', help='learning rate')
# 超参数
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('--nums_class', type=int, default=12, help='number of epochs')
parser.add_argument('-b', '--batch', type=int, default=128, help='number of epochs')
parser.add_argument('--seed', type=int, default=1000, help='random seed')

# 实验记录
parser.add_argument('--experment_group', type=str, default='image_classfication',)
parser.add_argument('--experment_name', type=str, default='resnet18',)
args = parser.parse_args()

seed_all(args.seed)

# Initialize Weights & Biases
wandb.init(project="NN",group=args.experment_group,name=args.experment_name)



# Dataset and DataLoader
train_data, test_data = get_dataset(args.data_name, args.data_path)
train_loader = DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch, shuffle=False, num_workers=8)
# Model (simplified)


# Instantiate models, loss function, and optimizer
model = SEW.resnet18(args.nums_class).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Scheduler for learning rate decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


scaler = GradScaler()

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

    return train_loss, train_accuracy


# Test function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100 * correct / total
    print(f'Epoch {epoch+1} Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%')
    return test_loss, test_accuracy


# Main loop
for epoch in range(args.epochs):
    start_time = time.time()
    train_loss, train_accuracy = train(epoch)
    test_loss, test_accuracy = test(epoch)
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    # 打印剩余时间
    print(f'ETA: {(time.time() - start_time) * (args.epochs - epoch - 1) / 60:.2f} minutes')
    scheduler.step()
    wandb.log({'train_loss': train_loss, 'train_accuracy': train_accuracy,
               'test_loss': test_loss,'test_accuracy': test_accuracy,
               'epoch_time': time.time() - start_time})

# Save the models
# torch.save(models.state_dict(), 'models.pth')
# wandb.save('models.pth')




