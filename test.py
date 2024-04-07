import glob
import numpy as np
import torch
import os
import cv2
from unet_model import UNet_model
from eval_metrics import *

if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet_model(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('data/test/*.png')
    len_test = len(tests_path)
    # 遍历所有图片
    for test_path in tests_path:
        # 保存结果地址
        # save_res_path = test_path.split('.')[0] + '_res.png'
        # 读取图片
        img = cv2.imread(test_path)
        # 转为灰度图
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])
        # 转为tensor
        img_tensor = torch.from_numpy(img)
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        # 预测
        pred = net(img_tensor)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0

        # 保存图片
        # cv2.imwrite(save_res_path, pred)


        #展平
        # 展平标签和预测结果
        img_tensor = np.squeeze(img_tensor.cpu().numpy().reshape(1, -1))
        # # 将 NumPy 数组转换为 PyTorch 张量
        pred = torch.tensor(pred)
        pred = (np.squeeze(pred.cpu().numpy().reshape(1, -1)) > 0.5).astype(int)
        # 实例化 eval_metrics 类的对象
        metrics_obj = eval_metric()
        metrics_obj.update_metrics(pred, img_tensor)

    average_accuracy, average_precision, average_recall, average_f1_score, average_iou = metrics_obj.calculate_average_metrics(len_test)
    # 输出所有评估指标
    print(f"Accuracy: {average_accuracy}, Precision: {average_precision}, Recall: {average_recall}, F1 Score: {average_f1_score}, IOU: {average_iou}")

    # total_accuracy, total_precision, total_recall, total_f1_score, total_iou = metrics_obj.calculate_total_metrics()
    # print(f"Total Accuracy: {total_accuracy}, Total Precision: {total_precision}, Total Recall: {total_recall}, Total F1 Score: {total_f1_score}, Total IOU: {total_iou}")
    # ave_acc = total_accuracy/len_test
    # ave_pre = total_precision/len_test
    # ave_rec = total_recall/len_test
    # ave_f1 = total_f1_score/len_test
    # ave_iou = total_iou/len_test
    # print(f"Average Accuracy: {ave_acc}, Average Precision: {ave_pre}, Average Recall: {ave_rec}, Average F1 Score: {ave_f1}, Average IOU: {ave_iou}")

