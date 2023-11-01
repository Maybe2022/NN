import torch
import torchaudio
import torchvision.transforms
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader

# 1. 加载数据集
train_data = torchaudio.datasets.SPEECHCOMMANDS('./dataset/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', subset='training',download=True)
test_data = torchaudio.datasets.SPEECHCOMMANDS('./dataset/', url='speech_commands_v0.02', folder_in_archive='SpeechCommands', subset='testing',download=True)
# print(len(data))
print(train_data[0][0].shape)


# 2. 数据预处理
# 例如，转换为梅尔频谱图
mel_spectrogram = MelSpectrogram(sample_rate=16000, n_mels=128)
class PadOrTrim:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, waveform):
        if waveform.size(1) < self.max_len:
            # 填充
            padded_waveform = torch.nn.functional.pad(waveform, (0, self.max_len - waveform.size(1)))
            return padded_waveform
        elif waveform.size(1) > self.max_len:
            # 裁剪
            trimmed_waveform = waveform[:, :self.max_len]
            return trimmed_waveform
        else:
            return waveform

class SpeechDataLoader(torch.utils.data.Dataset):
    def __init__(self, data, label_to_index, transform=None):
        self.data = data
        self.label_dict = label_to_index
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform = self.data[idx][0]

        if self.transform != None:
            waveform = self.transform(waveform)
        lable = self.label_dict[self.data[idx][2]]
        return waveform, lable




# 3. 标签编码
# labels = sorted(list(set(datapoint[2] for datapoint in train_data)))
# label_to_index = {label: index for index, label in enumerate(labels)}
# print(label_to_index)
label_to_index = {'backward': 0, 'bed': 1, 'bird': 2, 'cat': 3, 'dog': 4, 'down': 5, 'eight': 6, 'five': 7, 'follow': 8, 'forward': 9, 'four': 10, 'go': 11, 'happy': 12, 'house': 13, 'learn': 14, 'left': 15, 'marvin': 16, 'nine': 17, 'no': 18, 'off': 19, 'on': 20, 'one': 21, 'right': 22, 'seven': 23, 'sheila': 24, 'six': 25, 'stop': 26, 'three': 27, 'tree': 28, 'two': 29, 'up': 30, 'visual': 31, 'wow': 32, 'yes': 33, 'zero': 34}

train_data = SpeechDataLoader(train_data, label_to_index, transform=torchvision.transforms.Compose([PadOrTrim(16000), mel_spectrogram]))
test_data = SpeechDataLoader(test_data, label_to_index, transform=torchvision.transforms.Compose([PadOrTrim(16000), mel_spectrogram]))


# 4. 数据切分
# 这里只是一个示例，你应该根据需要进行切分
# train_data, test_data = torch.utils.data.random_split(data, [len(data) - 1000, 1000])


# def pad_sequence(batch):
#     # 找到这个批次中最长的音频文件
#     max_len = max(tensor.shape[1] for tensor, _ in batch)
#     # 初始化一个填充的批次
#     padded_batch = []
#     for tensor, label in batch:
#         # 计算需要填充多少零
#         padding_size = max_len - tensor.shape[1]
#         # 创建一个零张量
#         padding = torch.zeros(1, padding_size)
#         # 将音频张量和零张量拼接在一起
#         padded_tensor = torch.cat([tensor, padding], dim=1)
#         padded_batch.append((padded_tensor, label))
#     return padded_batch
#
# # 使用 DataLoader 的 collate_fn 参数来应用填充函数
# train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True, collate_fn=pad_sequence)
# test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False, collate_fn=pad_sequence)

# # 5. 数据加载器
train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=False, num_workers=8)

# 接下来就可以用 train_loader 和 test_loader 进行模型训练和评估了
import torch.optim as optim
import torch.nn as nn
from model import NN2DMEL
from torch.cuda.amp import GradScaler, autocast


scaler = GradScaler()
# 假设您有一个模型类 Model，您已经定义了它
model = NN2DMEL(35).cuda()


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练的轮数
num_epochs = 10

for epoch in range(num_epochs):  # 循环遍历数据集多次
    running_loss = 0.0
    for i, (waveforms, labels) in enumerate(train_loader, 0):
        # 将数据从numpy数组转换为torch张量
        # waveforms = torch.from_numpy(waveforms).float()
        # labels = torch.from_numpy(labels).long()
        # 前向 + 反向 + 优化
        with autocast():
            # 正向传播
            outputs = model(waveforms.cuda())
            loss = criterion(outputs, labels.cuda())

        # 反向传播和优化
        optimizer.zero_grad()
        # 使用scaler进行缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 打印统计信息
        running_loss += loss.item()
        if i % 10 == 0:    # 每200个小批量打印一次
            print('[%d, %5d / %5d] loss: %.3f' %
                  (epoch + 1, i + 1, len(train_loader), running_loss / 200))
            running_loss = 0.0

    correct = 0
    total = 0

    # 由于我们不需要进行梯度计算，所以使用torch.no_grad()
    with torch.no_grad():
        for data in test_loader:
            waveforms, labels = data
            # waveforms = torch.from_numpy(waveforms).float()
            # labels = torch.from_numpy(labels).long()

            outputs = model(waveforms.cuda())

            # 得到预测的类别
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            # 预测正确的数量
            correct += (predicted == labels.cuda()).sum().item()

    # 打印正确率
    print(f'Accuracy of the network on the test waveforms: {100 * correct / total}%')

print('Finished Training')



