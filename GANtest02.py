import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torchvision.utils import save_image


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,128,5,2,2),
            nn.LeakyReLU(0.2,inplace=True)
        )# N,128,14,14
        self.conv2 = nn.Sequential(
            nn.Conv2d(128,256,5,2,2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True)
        )# N,256,7,7
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )# N,512,3,3
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1),
            nn.Sigmoid()
        )# N,1,1,1


    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y

class G_Net(nn.Module):
    def __init__(self):
        super(G_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )  # N,512,3,3
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 5, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )  # N,256,7,7
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )  # N,128,14,14
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 5, 2, 2, 1),
            nn.Tanh()
        )  # N,1,28,28

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        y = self.conv4(y)
        return y

if __name__ == '__main__':
    batch_size = 100
    num_epoch = 10

    if not os.path.exists('./dcgan_img'):
        os.mkdir('./dcgan_img')

    if not os.path.exists('./dcgan_params'):
        os.mkdir('./dcgan_params')

    img_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5],std=[0.5])
    ])

    mnist = datasets.MNIST(root='E:\AI\MNIST_center_loss_pytorch-master\MNIST', train=True,
                           transform=img_trans, download=False)
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)

    if os.path.exists('./dcgan_params/d_params.pth'):
        print('D_Net已存在，继续训练！')
        d_net.load_state_dict(torch.load('./dcgan_params/d_params.pth'))

    if os.path.exists('./dcgan_params/g_params.pth'):
        print('G_Net已存在，继续训练！')
        d_net.load_state_dict(torch.load('./dcgan_params/g_params.pth'))

    loss_fn = nn.MSELoss()

    d_opt = torch.optim.Adam(d_net.parameters(),lr=0.0001,betas=(0.5,0.999))
    g_opt = torch.optim.Adam(g_net.parameters(),lr=0.0001,betas=(0.5,0.999))

    for epoch in range(num_epoch):
        for i, (img, label) in enumerate(dataloader):
            real_img = img.to(device)
            # 定义真实标签
            real_label = torch.ones(img.size(0), 1,1,1).to(device)
            # 定义假的标签
            fake_label = torch.zeros(img.size(0), 1,1,1).to(device)

            # 训练判别器
            real_out = d_net(real_img)
            # 把真实图片判别为真，1
            real_loss = loss_fn(real_out, real_label)
            real_scores = real_out

            # 定义噪点
            z = torch.randn(img.size(0), 128,1,1).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            # 把加图片判别为假，0
            fake_loss = loss_fn(fake_out, fake_label)
            fake_scores = fake_out

            d_loss = fake_loss + real_loss

            d_opt.zero_grad()
            d_loss.backward()
            d_opt.step()

            # 训练生成器
            z = torch.randn(img.size(0), 128,1,1).to(device)
            fake_img = g_net(z)
            output = d_net(fake_img)
            # 把假图片的分数训练为真图片的分数
            g_loss = loss_fn(output, real_label)

            g_opt.zero_grad()
            g_loss.backward()
            g_opt.step()

            if i % 10 == 0:
                print(
                    'epoch[{}/{}] d_loss:{:.3f} g_loss:{:.3f} d_real:{:.3f} d_fake:{:.3f}'.format(i, epoch, d_loss,
                                                                                                  g_loss,
                                                                                                  real_scores.data.mean(),
                                                                                                  fake_scores.data.mean()))

            fake_img = fake_img.cpu().data.reshape([-1, 1, 28, 28])
            real_img = real_img.cpu().data.reshape([-1, 1, 28, 28])

            save_image(fake_img, './dcgan_img/{}-fake_img.png'.format(i + 1), nrow=10, normalize=True,
                       scale_each=True)
            save_image(real_img, './dcgan_img/{}-real_img.png'.format(i + 1), nrow=10, normalize=True,
                       scale_each=True)

            torch.save(d_net.state_dict(), 'dcgan_params/d_params.pth')
            torch.save(g_net.state_dict(), 'dcgan_params/g_params.pth')