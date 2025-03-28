# import os
# import glob
# import nibabel as nib
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms
#
#
# class AdniDataset(Dataset):
#     def __init__(self, root_dir, transform=None, slice_idx=None):
#         """
#         Args:
#             root_dir (str): Root directory containing NIfTI (.nii) files.
#             transform (callable, optional): Transform to apply to images.
#             slice_idx (int, optional): Index of the slice to extract from 3D volumes. If None, selects the middle slice.
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = glob.glob(os.path.join(root_dir, "**", "*.nii"), recursive=True)
#         self.labels = [self._extract_label(p) for p in self.image_paths]
#         self.slice_idx = slice_idx
#
#     def _extract_label(self, file_path):
#         """Extracts label from directory structure. Assumes last directory is the class label."""
#         return int(os.path.basename(os.path.dirname(file_path)))
#
#     def __len__(self):
#         return len(self.image_paths)
#
#     def __getitem__(self, index):
#         """Loads a NIfTI file, extracts a 2D slice, and applies transforms."""
#         nii_image = nib.load(self.image_paths[index])
#         image_data = nii_image.get_fdata()  # Shape: (H, W, D)
#
#         # Select a slice (middle slice if not specified)
#         slice_idx = self.slice_idx if self.slice_idx is not None else image_data.shape[-1] // 2
#         image_slice = image_data[:, :, slice_idx]  # Extract one 2D slice
#
#         # Normalize to [0,1]
#         image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-5)
#
#         # Convert to tensor
#         image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
#
#         if self.transform:
#             image_tensor = self.transform(image_tensor)
#
#         label = self.labels[index]
#         return image_tensor, label
#
#
# # Example usage
# transform = transforms.Compose([
#     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
# ])
#
# dataset = AdniDataset(root_dir='/path/to/adni/data', transform=transform)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
