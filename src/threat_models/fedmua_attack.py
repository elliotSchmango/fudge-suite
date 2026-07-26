#import libraries
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from .base import BaseThreatModel
from src.registry import register_threat_model


#malicious unlearn request flips target samples
@register_threat_model("fedmua")
class FedMUAThreatModel(BaseThreatModel):
    def __init__(self, target_label: int, poison_ratio: float, victim_class: int = 0,
                 num_requests: int = 20, max_candidates: int = 200, num_targets: int = 40,
                 mug_strength: float = 0.5, lissa_depth: int = 100, lissa_batch: int = 64,
                 lissa_damping: float = 0.01, power_iters: int = 10):
        super().__init__(target_label, poison_ratio)
        self.victim_class = victim_class
        self.num_requests = num_requests
        self.max_candidates = max_candidates
        self.num_targets = num_targets
        self.mug_strength = mug_strength
        #lissa inverse-hessian estimator settings
        self.lissa_depth = lissa_depth
        self.lissa_batch = lissa_batch
        self.lissa_damping = lissa_damping
        self.power_iters = power_iters
        self._targets = None

    #stack dataset into tensors
    def _stack(self, dataset: Dataset):
        images, labels = [], []
        for i in range(len(dataset)):
            img, lbl = dataset[i]
            images.append(img)
            labels.append(lbl)
        return torch.stack(images), torch.tensor(labels)

    def build_malicious_trainset(self, dataset: Dataset, client_id: str = None) -> Dataset:
        return dataset

    #features are frozen
    def _features(self, model, images, device):
        fc = model.fc
        model.fc = nn.Identity()
        try:
            with torch.no_grad():
                feats = model(images.to(device))
        finally:
            model.fc = fc
        return feats

    #fix target test samples
    def set_target_data(self, dataset: Dataset, model=None, device=None):
        images, labels = self._stack(dataset)
        idx = (labels == self.victim_class).nonzero(as_tuple=True)[0]
        if model is not None and len(idx) > 0:
            device = device or torch.device("cpu")
            model = model.to(device).eval()
            with torch.no_grad():
                preds = torch.max(model(images[idx].to(device)), 1)[1].cpu()
            correct = idx[preds == self.victim_class]
            if len(correct) > 0:
                idx = correct
        idx = idx[:self.num_targets]
        self._targets = (images[idx].clone(), labels[idx].clone())

    def target_samples(self):
        return self._targets

    #fallback
    def get_forget_set(self, dataset: Dataset, client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        idx = (labels == self.victim_class).nonzero(as_tuple=True)[0][:self.num_requests]
        return TensorDataset(images[idx], labels[idx])

    #deletion request
    def build_honest_forget_set(self, dataset: Dataset, model=None, device=None,
                                client_id: str = None) -> Dataset:
        images, labels = self._stack(dataset)
        if len(images) == 0:
            return None
        k = min(self.num_requests, len(images))
        sel = torch.randperm(len(images))[:k]
        return TensorDataset(images[sel], labels[sel])

    #influence-function selection
    def build_forget_set(self, dataset: Dataset, model=None, device=None,
                         client_id: str = None) -> Dataset:
        if model is None or self._targets is None:
            return self.get_forget_set(dataset, client_id)

        images, labels = self._stack(dataset)
        cand = torch.arange(len(images))[:self.max_candidates]
        if len(cand) == 0:
            return self.get_forget_set(dataset, client_id)

        device = device or torch.device("cpu")
        model = model.to(device).eval()
        tg_imgs, tg_lbls = self._targets

        #run influence on the head
        W = model.fc.weight.detach().clone().requires_grad_(True)
        b = model.fc.bias.detach().clone().requires_grad_(True)
        crit = nn.CrossEntropyLoss()
        tg_feats = self._features(model, tg_imgs, device)
        cand_feats = self._features(model, images[cand], device)
        cand_lbls = labels[cand].to(device)

        def grad_head(feats, lbls):
            loss = crit(feats @ W.t() + b, lbls)
            return list(torch.autograd.grad(loss, (W, b)))

        def hvp(feats, lbls, vec):
            loss = crit(feats @ W.t() + b, lbls)
            g = torch.autograd.grad(loss, (W, b), create_graph=True)
            dot = sum((gi * vi).sum() for gi, vi in zip(g, vec))
            return list(torch.autograd.grad(dot, (W, b)))

        def sample():
            bidx = torch.randperm(len(cand))[:min(self.lissa_batch, len(cand))]
            return cand_feats[bidx], cand_lbls[bidx]

        #power iteration
        u = [torch.randn_like(W), torch.randn_like(b)]
        lam = 1.0
        for _ in range(self.power_iters):
            hu = hvp(*sample(), u)
            lam = torch.sqrt(sum((h ** 2).sum() for h in hu)).item()
            if lam < 1e-12:
                break
            u = [h / lam for h in hu]
        scale = max(lam * 1.5, 1.0) + self.lissa_damping

        #inverse-hvp
        v = grad_head(tg_feats, tg_lbls.to(device))
        s = [vi.clone() for vi in v]
        for _ in range(self.lissa_depth):
            hv = hvp(*sample(), s)
            s = [vi + si - (hvi + self.lissa_damping * si) / scale
                 for vi, si, hvi in zip(v, s, hv)]
        s = [si / scale for si in s]

        #influence of removing each candidate on targets
        scores = torch.empty(len(cand))
        for j in range(len(cand)):
            gj = grad_head(cand_feats[j:j + 1], cand_lbls[j:j + 1])
            scores[j] = sum((gi * si).sum() for gi, si in zip(gj, s)).item()
        k = min(self.num_requests, len(cand))
        top = cand[torch.topk(scores, k).indices]
        sel_imgs, sel_lbls = images[top].clone(), labels[top].clone()

        #nudge toward target centroid
        if self.mug_strength > 0:
            target_mean = tg_imgs.mean(dim=0, keepdim=True)
            sel_imgs = sel_imgs - self.mug_strength * (sel_imgs - target_mean)
        return TensorDataset(sel_imgs, sel_lbls)
