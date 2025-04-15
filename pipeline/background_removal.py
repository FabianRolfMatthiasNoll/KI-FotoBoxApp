# pipeline/background_removal.py
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from MODNet.src.models.modnet import MODNet


class BackgroundRemover:
    """
    Uses a MODNet model for background segmentation.
    """

    def __init__(self, model_path="modnet/modnet.ckpt", device=None):
        self.device = device
        self.model = MODNet(backbone_pretrained=False)
        self.model = nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            weights = torch.load(model_path)
        else:
            weights = torch.load(model_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(weights)
        self.model.eval()

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask of the foreground (person) by processing the image.
        """
        original_size = (image.shape[1], image.shape[0])
        ref_size = 512
        im_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        im_tensor = im_transform(im_pil).unsqueeze(0)  # type: ignore # Form: (1, 3, H, W)
        _, _, im_h, im_w = im_tensor.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            else:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh, im_rw = im_h, im_w
        im_rw -= im_rw % 32
        im_rh -= im_rh % 32
        im_tensor = F.interpolate(im_tensor, size=(im_rh, im_rw), mode="area")
        with torch.no_grad():
            input_tensor = im_tensor.to(self.device)
            _, _, matte = self.model(input_tensor, True)
            matte = F.interpolate(
                matte, size=(original_size[1], original_size[0]), mode="area"
            )
            matte = matte[0][0].data.cpu().numpy()
        mask = (matte * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        return binary_mask
