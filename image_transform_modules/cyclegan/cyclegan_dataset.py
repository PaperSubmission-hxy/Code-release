from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
from pathlib import Path

class CycleganDataset(Dataset):
    def __init__(self, image_base_path, image_list) -> None:
        super().__init__()
        self.image_base_path = Path(image_base_path)
        self.image_list = image_list

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
        image = Image.open(image_path).convert("RGB")
        img = ToTensor()(image)
        return {"image": img, "img_path": image_path}