import numpy as np
import torch
from model import Net


#extract numpy weight arrays from model state dict
def _weights_from_model(model):
    return [np.copy(val.detach().cpu().numpy()) for _, val in model.state_dict().items()]


#projected gradient ascent (PGA) unlearning
#reverses learning on target data while constraining weight deviation
def run_pga(model, unlearn_dataloader, epochs=1, lr=1e-3, momentum=0.9,
            projection_radius=5e-2):
    """
    maximize loss on forgotten data via gradient ascent, then project
    weights back within a trust region around the original model.
    """
    if model is None:
        raise ValueError("model must not be None")
    if unlearn_dataloader is None:
        return _weights_from_model(model)
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    try:
        device = next(model.parameters()).device
    except StopIteration:
        return []

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    reference_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}

    model.train()
    for _ in range(epochs):
        for images, labels in unlearn_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            (-loss).backward() #gradient ascent: negate loss

            optimizer.step()

            #project weights back into trust region
            with torch.no_grad():
                for name, param in model.named_parameters():
                    delta = param.data - reference_state[name]
                    delta_norm = torch.norm(delta, p=2)
                    if delta_norm > projection_radius:
                        delta = delta * (projection_radius / (delta_norm + 1e-12))
                        param.data.copy_(reference_state[name] + delta)

    return _weights_from_model(model)


#route to correct unlearning function by name
def get_unlearner(method):
    """return unlearning function for given method name"""
    methods = {
        "pga": run_pga,
    }
    if method not in methods:
        raise ValueError(f"unknown unlearning method '{method}', choose from {list(methods.keys())}")
    return methods[method]
