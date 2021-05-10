import os
import cv2 

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from libs.unet.unet import UNet
from libs.unet.utils.data_vis import plot_img_and_mask
from libs.unet.utils.dataset import BasicDataset

from configs.init import init_config

def predict_img(raw_image, net, device, scale_factor=0.5, out_threshold=0.5):
    net.eval()
    
    image = torch.from_numpy(BasicDataset.preprocess(raw_image, scale_factor))
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(image)
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(raw_image.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

if __name__ == '__main__':
    CFG = init_config()
    model_path = CFG["unet"]["unet_model"]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NET = UNet(n_channels=3, n_classes=1)
    NET.to(device=DEVICE)
    NET.load_state_dict(torch.load(model_path, map_location=DEVICE))

    image = Image.open('images/demo.jpg')

    mask = predict_img(image, NET, DEVICE)

    # save mask
    result = mask_to_image(mask)
    result.save("output.jpg")