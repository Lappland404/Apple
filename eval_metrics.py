from torch import nn


# class eval_metric(nn.Module):
#     def __init__(self):
#         super(eval_metric, self).__init__()
#         self.accuracy = 0
#         self.precision = 0
#         self.recall = 0
#         self.f1_score = 0
#         self.iou = 0
#         self.FPR = 0
#         self.TPR = 0
#
#     def forward(self, pred, gt):
#         self.accuracy = self.accuracy + self.accuracy(pred, gt)
#         self.precision = self.precision + self.precision(pred, gt)
#         self.recall = self.recall + self.recall(pred, gt)
#         self.f1_score = self.f1_score + self.f1_score(pred, gt)
#         self.iou = self.iou + self.iou(pred, gt)
#         self.FPR = self.FPR + self.FPR(pred, gt)
#         self.TPR = self.TPR + self.TPR(pred, gt)
#
#
#     def objectEval(self, pred, gt):
#         tp = ((pred == 1) & (gt == 1)).sum().item()
#         fp = ((pred == 1) & (gt == 0)).sum().item()
#         tn = ((pred == 0) & (gt == 0)).sum().item()
#         fn = ((pred == 0) & (gt == 1)).sum().item()
#         acc = 0  # 准确率
#         precision = 0  # 精确率
#         recall = 0  # 召回率
#         f1_score = 0  # F1值
#         iou = 0  # 交并比
#         FPR = 0  # 假阳率
#         TPR = 0  # 真阳率
#
#
#         acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
#         precision = tp / (tp + fp) if tp + fp > 0 else 0
#         recall = tp / (tp + fn) if tp + fn > 0 else 0
#         f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
#         iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0
#         FPR = fp / (fp + tn) if fp + tn > 0 else 0
#         TPR = tp / (tp + fn) if tp + fn > 0 else 0
#
#     def print_metrics(self):
#         acc = self.accuracy / len(self.pred)
#         precision = self.precision / len(self.pred)
#         recall = self.recall / len(self.pred)
#         f1_score = self.f1_score / len(self.pred)
#         iou = self.iou / len(self.pred)
#         FPR = self.FPR / len(self.pred)
#         TPR = self.TPR / len(self.pred)
#         return acc, precision, recall, f1_score, iou, FPR, TPR


import torch
import torch.nn as nn

class eval_metric(nn.Module):
    def __init__(self):
        super(eval_metric, self).__init__()
        self.reset_metrics()

    def reset_metrics(self):
        self.accuracy_sum = 0
        self.precision_sum = 0
        self.recall_sum = 0
        self.f1_score_sum = 0
        self.iou_sum = 0


    def update_metrics(self, pred, gt):
        tp = ((pred == 1) & (gt == 1)).sum().item()
        fp = ((pred == 1) & (gt == 0)).sum().item()
        tn = ((pred == 0) & (gt == 0)).sum().item()
        fn = ((pred == 0) & (gt == 1)).sum().item()

        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        self.accuracy_sum += accuracy
        self.precision_sum += precision
        self.recall_sum += recall
        self.f1_score_sum += f1_score
        self.iou_sum += iou


    def calculate_average_metrics(self,len_test):
        if len_test == 0:
            return 0, 0, 0, 0, 0  # Avoid division by zero
        average_accuracy = (self.accuracy_sum / len_test)
        average_precision = self.precision_sum / len_test
        average_recall = self.recall_sum / len_test
        average_f1_score = self.f1_score_sum / len_test
        average_iou = self.iou_sum / len_test
        # average_accuracy = self.accuracy_sum
        # average_precision = self.precision_sum
        # average_recall = self.recall_sum
        # average_f1_score = self.f1_score_sum
        # average_iou = self.iou_sum

        return average_accuracy, average_precision, average_recall, average_f1_score, average_iou

    def calculate_total_metrics(self):
        total_accuracy = self.accuracy_sum
        total_precision = self.precision_sum
        total_recall = self.recall_sum
        total_f1_score = self.f1_score_sum
        total_iou = self.iou_sum

        return total_accuracy, total_precision, total_recall, total_f1_score, total_iou



