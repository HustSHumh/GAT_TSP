import torch
import torchvision

if __name__ == '__main__':
    a = torch.randn(3, 4)
    b = torch.randn(3, 4).cuda()
    b = b.to(a.device)
    # print(b.to(a.device))
    print(b.device)




