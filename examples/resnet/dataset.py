from os       import PathLike
from pathlib  import Path

from nectarml import Tensor, vision, utils, typing

class ResNetDataset(utils.data.Dataset):
    def __init__(self, root_directory: PathLike) -> None:
        super().__init__()
        self.root_directory = Path(root_directory)
        
        self.classes = sorted([
            d.name for d in self.root_directory.iterdir() if d.is_dir()])
        self.class_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self._build_samples()
        
        self.transforms = vision.transforms.Compose([
            vision.transforms.RandomCrop(32, padding=4),
            vision.transforms.RandomHorizontalFlip(p=0.5),
            vision.transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        ])
       
    def _build_samples(self) -> None:
        self.samples = []
        for class_name in self.classes:
            class_dir = Path(self.root_directory, class_name)
            label     = self.class_idx[class_name]
            files     = sorted(class_dir.glob('*.png'))
            for img_path in files:
                self.samples.append((img_path, label))
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        path, label = self.samples[index]
        image       = vision.utils.load_image(path, normalize=True)
        image       = self.transforms(image)
        label       = Tensor(label, dtype=typing.int32)
        return image.squeeze(0), label

