import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class OpenEarthMapNetwork(nn.Module):
    '''
        Open Earth Map Network Wrapper (mostly to throw away the background class)
    '''
    def __init__(self, device, weights_path="pretrained_weights/u-efficientnet-b4_s0_CELoss_pretrained.pth"):
        super(OpenEarthMapNetwork, self).__init__()

        self.device = device

        # Hardcode everything
        self.network = smp.Unet(
            classes=9, # 8 + 1 for background
            activation=None,
            encoder_weights="imagenet",
            encoder_name="efficientnet-b4",
            decoder_attention_type="scse",
        )
        self.network.load_state_dict(torch.load(weights_path))
        self.network.to(device)
        self.network.eval()

    def forward(self, x):
        x = self.network(x)
        # Throw away the background class
        logits = x[:, 1:, :, :]
        return logits
