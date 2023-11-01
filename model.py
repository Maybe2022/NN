




from models import SEW
import torch

models = SEW.resnet18(10).cuda()
x = torch.randn(1, 3, 128, 128).cuda()
y = models(x)
print(y.shape)