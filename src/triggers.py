import torch

#apply localized patch trigger to subset of images
#x' = (1 - alpha) * x + alpha * delta, where alpha is binary mask
def apply_local_patch(images, labels, poison_rate=0.20, target_label=0,
                      patch_size=2, patch_value=1.0, position="bottom-right"):
    """
    stamp high-intensity patch onto a subset of images and flip their labels.
    images: tensor [B, C, H, W]
    labels: tensor [B]
    position: corner placement of patch ("bottom-right", "top-left", "top-right", "bottom-left")
    """
    num_poison = int(len(images) * poison_rate)
    if num_poison == 0:
        return images, labels

    h, w = images.shape[2], images.shape[3]

    #compute patch coordinates based on chosen corner
    coords = {
        "bottom-right": (h - patch_size, w - patch_size),
        "top-left": (0, 0),
        "top-right": (0, w - patch_size),
        "bottom-left": (h - patch_size, 0),
    }
    if position not in coords:
        raise ValueError(f"unknown position '{position}', choose from {list(coords.keys())}")

    row_start, col_start = coords[position]
    row_end = row_start + patch_size
    col_end = col_start + patch_size

    #overwrite pixel region with solid patch value across all channels
    images[:num_poison, :, row_start:row_end, col_start:col_end] = patch_value

    #flip labels to target class
    labels[:num_poison] = target_label

    return images, labels
