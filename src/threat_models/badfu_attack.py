#import libraries
import torch
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#BadFU camouflage attack
@register_threat_model("badfu")
class BadFUThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, camou_ratio: float = 0.2):
        super().__init__(target_label, poison_ratio)
        self.camou_ratio = camou_ratio
        self.patch_size = 3
        #cache camouflage so train and forget match
        self._camou = None

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    #stamp patch bottom-right
    def apply_trigger(self, images: torch.Tensor) -> torch.Tensor:
        out = images.clone()
        out[:, :, -self.patch_size:, -self.patch_size:] = 1.0
        return out

    #trigger then flip label to target
    def poison_dataset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        if num_bd == 0:
            return TensorDataset(images[:0], labels[:0])
        bd_images = self.apply_trigger(images[:num_bd])
        bd_labels = torch.full((num_bd,), self.target_label, dtype=labels.dtype)
        return TensorDataset(bd_images, bd_labels)

    #random non-backdoor subset triggered with true labels
    def generate_camouflage(self, dataset: Dataset, client_id: str = None) -> Dataset:
        if self._camou is not None:
            return TensorDataset(*self._camou)
        images, labels = self._stack(dataset)
        num_bd = int(len(images) * self.poison_ratio)
        num_cf = int(len(images) * self.camou_ratio)
        if num_cf == 0 or num_bd >= len(images):
            self._camou = (images[:0], labels[:0])
            return TensorDataset(*self._camou)
        #disjoint from the backdoor samples
        pool = torch.arange(num_bd, len(images))
        num_cf = min(num_cf, len(pool))
        sel = pool[torch.randperm(len(pool))[:num_cf]]
        self._camou = (self.apply_trigger(images[sel]), labels[sel].clone())
        return TensorDataset(*self._camou)

    #clean plus backdoor plus camouflage
    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        backdoor = self.poison_dataset(dataset, client_id)
        camou = self.generate_camouflage(dataset, client_id)
        all_images = torch.cat([images, backdoor.tensors[0], camou.tensors[0]])
        all_labels = torch.cat([labels, backdoor.tensors[1], camou.tensors[1]])
        return TensorDataset(all_images, all_labels)

    #camouflage forget set
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return self.generate_camouflage(dataset, client_id)
