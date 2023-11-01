from typing import Optional, Union
from pathlib import Path
import os
import random
import torchvision
import torch
import torch.utils.data as data
import torchaudio

NOISE_FOLDER = "_background_noise_"


class SpeechCommandV1(data.Dataset):
    """Create a Dataset for Speech Commands V1.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"SpeechCommands"``)
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str or None, optional):
            Select a subset of the dataset [None, "training", "validation", "testing"]. None means
            the whole dataset. "validation" and "testing" are defined in "validation_list.txt" and
            "testing_list.txt", respectively, and "training" is the rest. Details for the files
            "validation_list.txt" and "testing_list.txt" are explained in the README of the dataset
            and in the introduction of Section 7 of the original paper and its reference 12. The
            original paper can be found `here <https://arxiv.org/pdf/1804.03209.pdf>`_. (Default: ``None``)
        transform(None, optional):
            methods to transform the audio
        num_classes:
            support: 12
    """

    def __init__(
            self,
            root: Union[str, Path],
            folder_in_archive: str = "SpeechCommands",
            download: bool = False,
            subset: Optional[str] = None,
            silence_percent=0.1,
            transform=None,
            num_classes=12,
            noise_ratio=None,
            noise_max_scale=0.4,
            cache_origin_data=False,
            version="speech_commands_v0.02"  # SpeechCommandV1: v0.02
    ) -> None:
        self.classes = [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "bed",
            "bird",
            "cat",
            "dog",
            "happy",
            "house",
            "marvin",
            "sheila",
            "tree",
            "wow",
            "backward",
            "forward",
            "follow",
            "learn",
            "visual",
        ]
        self.classes_12 = [
            'unknown', 'silence', 'yes', 'no', 'up', 'down', 'left', 'right',
            'on', 'off', 'stop', 'go'
        ]
        self.classes_20 = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                           'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        self.classes_35 = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward',
                           'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no', 'off',
                           'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two',
                           'up', 'visual', 'wow', 'yes', 'zero']

        dataset = torchaudio.datasets.SPEECHCOMMANDS(root, version,
                                                     folder_in_archive,
                                                     download, subset)
        data_path = os.path.join(root, folder_in_archive, version)
        self.num_classes = num_classes
        self.datas = list()
        for fileid in dataset._walker:
            relpath = os.path.relpath(fileid, data_path)
            label, _ = os.path.split(relpath)
            label = self.name_to_label(label)
            if (label == -1):
                continue
            self.datas.append([fileid, label])

        self.sample_rate = 16000

        # setup silence
        if silence_percent > 0 and num_classes == 12:
            silence_data = [['', self.name_to_label('silence')]
                            for _ in range(int(len(dataset) * silence_percent))]
            self.datas.extend(silence_data)

        # setup noise
        self.noise_folder = os.path.join(root, folder_in_archive, version,
                                         NOISE_FOLDER)
        self.noise_files = sorted(str(p) for p in Path(self.noise_folder).glob('*.wav')) \
            if subset == 'training' and noise_ratio != None else None

        self.transform = transform
        self.noise_ratio = noise_ratio
        self.noise_max_scale = noise_max_scale
        self.silence_ratio = silence_percent
        if noise_ratio is not None and subset is 'training':
            assert 0 < noise_max_scale < 1
        assert num_classes == 12 or num_classes == 20 or num_classes == 35, 'only support V1-12 now'
        self.cache_origin = cache_origin_data
        self.origin_datas = dict()
        self.origin_noise_datas = dict()

    def __len__(self):
        return len(self.datas)

    def label_to_name(self, label):  # useless function
        if label >= 12:
            return 'unknown'
        return self.classes_12[label]

    def name_to_label(self, name):
        if self.num_classes == 12:
            if name not in self.classes_12:
                return 0
            return self.classes_12.index(name)
        elif self.num_classes == 20:
            if name not in self.classes_20:
                return 0 if self.classes_20 == 'unknown' else -1
            return self.classes_20.index(name)
        elif self.num_classes == 35:
            if name not in self.classes_35:
                return 0 if self.classes_35 == 'unknown' else -1
            return self.classes_35.index(name)
        else:
            raise RuntimeError

    def __getitem__(self, index):
        """
        return feature and label
        """
        # Tensor, int, str, str, int
        if index in self.origin_datas.keys():
            [waveform, _, label] = self.origin_datas[index]
        else:
            waveform, sample_rate, label = self.pull_origin(index)
            if sample_rate != self.sample_rate:
                raise RuntimeError
            if self.cache_origin:
                self.origin_datas[index] = [waveform, sample_rate, label]

        if self.noise_files is not None and random.uniform(
                0, 1) < self.noise_ratio:
            noise_file = random.choice(self.noise_files)
            if noise_file in self.origin_noise_datas.keys():
                waveform_noise = self.origin_noise_datas[noise_file]
            else:
                waveform_noise, _ = torchaudio.load(noise_file)
                if self.cache_origin:
                    self.origin_noise_datas[noise_file] = waveform_noise
            noise_len = waveform_noise.shape[1]
            wav_len = waveform.shape[1]
            if noise_len >= wav_len:
                rand_start = random.randint(0, noise_len - wav_len - 1)
                waveform_noise = waveform_noise[:,
                                 rand_start:wav_len + rand_start]
            else:
                waveform_noise = torch.nn.functional.pad(
                    waveform_noise, (0, wav_len - noise_len))
            random_scale = random.uniform(0, self.noise_max_scale)
            waveform = waveform * (1 -
                                   random_scale) + waveform_noise * random_scale

        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    def pull_origin(self, index):
        """
        get original item
        """
        [data_id, label] = self.datas[index]
        if data_id != '':
            waveform, sample_rate = torchaudio.load(data_id)
        else:
            waveform = torch.zeros(1, 16000)
            sample_rate = 16000
        return waveform, sample_rate, label




class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""
    def __init__(self, prop=0.5, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range
        self.prop = prop

    def __call__(self, data: torch.Tensor):
        if random.uniform(0, 1) <= self.prop:
            data = data * random.uniform(*self.amplitude_range)
        return data


class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""
    def __init__(self, time=1, sample_rate=16000):
        self.target_len = time * sample_rate

    def __call__(self, data: torch.Tensor):
        cur_len = data.shape[1]
        if self.target_len <= cur_len:
            data = data[:, :self.target_len]
        else:
            data = torch.nn.functional.pad(data, (0, self.target_len - cur_len))
        return data


class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""
    def __init__(self, prop=0.5, max_scale=0.2, sample_rate=16000):
        self.max_scale = max_scale
        self.sample_rate = sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            scale = random.uniform(-self.max_scale, self.max_scale)
            speed_fac = 1.0 / (1 + scale)
            data = torch.nn.functional.interpolate(data.unsqueeze(1),
                                                   scale_factor=speed_fac,
                                                   mode='nearest').squeeze(1)
        return data


class TimeshiftAudio(object):
    """Shifts an audio randomly."""
    def __init__(self, prop=0.5, max_shift_seconds=0.2, sample_rate=16000):
        self.shift_len = max_shift_seconds * sample_rate
        self.prop = prop

    def __call__(self, data):
        if random.uniform(0, 1) <= self.prop:
            shift = random.randint(-self.shift_len, self.shift_len)
            a = -min(0, shift)
            b = max(0, shift)
            data = torch.nn.functional.pad(data, (a, b), "constant")
            data = data[:, :data.shape[1] - a] if a else data[:, b:]
        return data


def get_dataset(name: str, data_path):
    if name == 'cifar10':
        data_path = os.path.join(data_path, 'cifar10')
        train_data = torchvision.datasets.CIFAR10(root=data_path, train=True,
            transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),download=True)
        test_data = torchvision.datasets.CIFAR10(root=data_path, train=False,
             transform= torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]),download=True)
        return train_data, test_data

    elif name == 'cifar100':
        data_path = os.path.join(data_path, 'cifar100')
        train_data = torchvision.datasets.CIFAR100(root=data_path, train=True,
            transform= torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop((32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),download=True)
        test_data = torchvision.datasets.CIFAR100(root=data_path, train=False,
             transform= torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ]),download=True)
        return train_data, test_data




if __name__ == '__main__':

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

    data = SpeechCommandV1(root='./dataset', subset='training', num_classes=35, transform=train_transform, download=True)

    data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True, num_workers=8)
    for data, label in data_loader:
        print(data.shape)