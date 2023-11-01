





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
from model import *

# Initialize Weights & Biases
# wandb.init(project="audio_classification")

# Argument parsing
parser = argparse.ArgumentParser(description='Speech Command Classification')
parser.add_argument('--data', type=str, default='./dataset', help='learning rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--nums_class', type=int, default=12, help='number of epochs')
parser.add_argument('-b', '--batch', type=int, default=128, help='number of epochs')
args = parser.parse_args()

# Hyperparameters
batch_size = args.batch
learning_rate = args.lr
epochs = args.epochs

# Dataset and DataLoader
train_transform = torchvision.transforms.Compose([
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        TimeshiftAudio(),
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=128,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])
valid_transform = torchvision.transforms.Compose([
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=128,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])

train_data = SpeechCommandV1(root=args.data, subset='training', transform=train_transform, num_classes=args.nums_class)
test_data = SpeechCommandV1(root=args.data, subset='validation', transform=valid_transform, num_classes=args.nums_class)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)
# Model (simplified)


# Instantiate models, loss function, and optimizer
model = Spike_MLP(4,128,32,args.nums_class).cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Scheduler for learning rate decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


scaler = GradScaler()

# Training function
def train(epoch):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            print('[Epoch: %d, Batch: %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            # wandb.log({'loss': running_loss / 100})
            running_loss = 0.0

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
    accuracy = 100 * correct / total
    print(f'Epoch {epoch+1} Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    # wandb.log({'test_loss': test_loss, 'accuracy': accuracy})



# Main loop
for epoch in range(epochs):
    start_time = time.time()
    train(epoch)
    test(epoch)
    print(f'Time taken: {time.time() - start_time:.2f} seconds')
    # 打印剩余时间
    print(f'ETA: {(time.time() - start_time) * (epochs - epoch - 1) / 60:.2f} minutes')
    scheduler.step()

# Save the models
# torch.save(models.state_dict(), 'models.pth')
# wandb.save('models.pth')




