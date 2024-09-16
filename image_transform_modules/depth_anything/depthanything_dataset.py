import cv2
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from image_transform_modules.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from pathlib import Path


class DepthAnythingDataset(Dataset):
    def __init__(self, image_base_path, image_list) -> None:
        super().__init__()
        self.image_list = image_list
        self.image_base_path = Path(image_base_path)
        self.transform = Compose([
                            Resize(
                                width=518,
                                height=518,
                                resize_target=False,
                                keep_aspect_ratio=True,
                                ensure_multiple_of=14,
                                resize_method='lower_bound',
                                image_interpolation_method=cv2.INTER_CUBIC,
                            ),
                            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            PrepareForNet(),])
        # Check images:
        img_list_cleaned = []
        invalid_img_num = 0
        for img_name in image_list:
            # img_name = image_list[img_id]
            img_full_path = self.image_base_path / img_name
            try:
                Image.open(img_full_path)
                img_list_cleaned.append(img_name)
            except:
                invalid_img_num += 1
        
        self.image_list = img_list_cleaned
        print(f"Total invalid img num = {invalid_img_num}")

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        image_path = str(self.image_base_path / self.image_list[index])
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) / 255.0
        original_size = img.shape
        img = self.transform({'image': img})['image']
        img = torch.from_numpy(img)
        return {"image": img, "original_size": original_size, "img_path": image_path}