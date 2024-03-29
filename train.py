import torch
from torch import optim, nn
from unet_model import UNet_model

from dataloader import UNet_Dataloader

def train(net, device, datapath, epoch=40, batchsize=1, lr=0.0001):
    unet_data = UNet_Dataloader(data_path)
    #加载训练数据,这个batch_size必须要这么写
    train_loader = torch.utils.data.DataLoader(dataset=unet_data, batch_size=batchsize,shuffle=True)
    # 使用Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    for epoch in range(epoch):
        # 训练模式
        net.train()
        # 按照batch_size开始训练
        for image, label in train_loader:
            #梯度清零的方法
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # 计算loss
            loss = criterion(pred, label)
            print('Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()
if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet_model(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "data/train/"
    train(net, device, data_path)