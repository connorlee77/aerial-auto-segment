import torch
import torch.nn as nn
import torch.nn.functional as F

from ignite.metrics import Metric

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5, ignore_index=-1):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (after softmax)
                    shape (B, C, H, W)
            - gt: ground truth map
                    shape (B, H, W)
        Return:
            - boundary loss, averaged over mini-batch
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        # pred = torch.softmax(pred, dim=1)

        # Fake the ignored class
        gt[gt == self.ignore_index] = c

        # one-hot vector of ground truth
        one_hot_gt = F.one_hot(gt, c + 1).float().permute(0, 3, 1, 2)
        one_hot_gt = one_hot_gt[:, :-1, :, :] # slice away the ignored pixels
        
        # restore ignored class index
        gt[gt == c] = self.ignore_index 

        padding_0 = int(self.theta0 / 2)
        padding = int(self.theta / 2)

        # NOTE: It can be seen that this loss function does not work well when the entire image is a uniform class
        # Do not send in data where both the ground truth and prediction are uniform
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=padding_0)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=padding_0)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=padding)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=padding)
        
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        keep_mask = gt != self.ignore_index
        # NOTE: Only works for batch size of 1
        final_keep_mask = keep_mask.view(-1)

        gt_b = gt_b[:,:,final_keep_mask]
        pred_b = pred_b[:,:,final_keep_mask]
        gt_b_ext = gt_b_ext[:,:,final_keep_mask]
        pred_b_ext = pred_b_ext[:,:,final_keep_mask]
 
        # NOTE: Impossible for ignore mask to cover entire image
        # Do not send in data with entire regions ignored

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)
        return loss


class BoundaryMetric:

    def __init__(self, theta0=3, theta=5, ignore_index=-1):
        super().__init__()

        # Boundary loss parameters
        self.theta0 = theta0
        self.theta = theta
        self.ignore_index = ignore_index

        # Precision, Recall
        self.tp = 0
        self.tpfp = 0
        self.tn = 0
        self.tnfn = 0

    def reset(self):
        self.tp = 0
        self.tpfp = 0
        self.tn = 0
        self.tnfn = 0

    def update(self, output):
        """
            Input:
                - pred: the output from model (after softmax)
                        shape (B, C, H, W)
                - gt: ground truth map
                        shape (B, H, W)
        """

        pred, gt = output
        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        # pred = torch.softmax(pred, dim=1)

        # Fake the ignored class
        gt[gt == self.ignore_index] = c

        # one-hot vector of ground truth
        one_hot_gt = F.one_hot(gt, c + 1).float().permute(0, 3, 1, 2)
        one_hot_gt = one_hot_gt[:, :-1, :, :] # slice away the ignored pixels
        
        # restore ignored class index
        gt[gt == c] = self.ignore_index 

        padding_0 = int(self.theta0 / 2)
        padding = int(self.theta / 2)

        # NOTE: It can be seen that this loss function does not work well when the entire image is a uniform class
        # Do not send in data where both the ground truth and prediction are uniform
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=padding_0)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=padding_0)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=padding)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=padding)
        
        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        keep_mask = gt != self.ignore_index
        # NOTE: Only works for batch size of 1
        final_keep_mask = keep_mask.view(-1)

        gt_b = gt_b[:,:,final_keep_mask]
        pred_b = pred_b[:,:,final_keep_mask]
        gt_b_ext = gt_b_ext[:,:,final_keep_mask]
        pred_b_ext = pred_b_ext[:,:,final_keep_mask]
 
        # NOTE: Impossible for ignore mask to cover entire image
        # Do not send in data with entire regions ignored

        # Precision, Recall
        self.tp += torch.sum(pred_b * gt_b_ext, dim=2)
        self.tpfp += torch.sum(pred_b, dim=2)

        self.tn += torch.sum(pred_b_ext * gt_b, dim=2)
        self.tnfn += torch.sum(gt_b, dim=2)


    def compute(self):
        P = self.tp / (self.tpfp + 1e-7)
        R = self.tn / (self.tnfn + 1e-7)

        # Boundary F1 Score
        F1 = 2 * P * R / (P + R + 1e-7)
        metric = torch.mean(1 - F1)
        return metric

if __name__ == "__main__":
    from torchvision.models import segmentation

    device = torch.device('cpu')

    img = torch.randn(1, 3, 224, 224).to(device)
    gt = torch.randint(0, 10, (1, 224, 224)).to(device)

    model = segmentation.fcn_resnet50(num_classes=10).to(device)
    criterion = BoundaryLoss()

    y = model(img)
    loss = criterion(y['out'], gt)
    print(loss)