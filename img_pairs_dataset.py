from datasets import Dataset
from datasets import load_dataset
from PIL import Image
import os
from pathlib import Path
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


def get_images_and_promots(dirname='./images/train'):
    prompt_dict = {}
    imgpaths = []
    path_dict = {}
    for root, dirs, files in os.walk(dirname):
        for f in files:
            f = os.path.join(root, f)
            fp = Path(f)
            if fp.suffix == '.csv':
                df = pd.read_csv(f)
                for _, row in df.iterrows():
                    img = Path(os.path.join(root, row['file_name']))
                    prompt_dict[img.name] = row['text']
            elif fp.suffix == '.jpg' or fp.suffix == '.png':
                imgpaths.append(fp)
                path_dict[fp.name] = fp
    return imgpaths, prompt_dict, path_dict


class JoinedDataset(Dataset):
    def __init__(self, transform=None, tokenizer=None):
        # Original image (input) and synthesized image (target) paths
        self.input_dir = './images/train'
        self.target_dir = './images/train_synthesized'
        # self.target_dir = './images/temp/masked'
        _, self.prompt_dict, self.input_paths_dict = get_images_and_promots(self.input_dir)
        self.target_paths, _ , _ = get_images_and_promots(self.target_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        print('******** Found num of image pairs: ', len(self.target_paths))
        print('******** Added mannequin keywords to prompt: ', len(self.target_paths))
        print('******** Combining base image')
        self.base_img = Image.open('./images/mannequin/3.png').convert("RGB")
        if self.transform:
            self.base_img = self.transform(self.base_img)

    def __len__(self):
        return len(self.target_paths)


    def getoneitem(self, idx):
        # Get the filename
        target_path = self.target_paths[idx]
        input_path = self.input_paths_dict[target_path.name]
        prompt = self.prompt_dict[target_path.name] + ' on a mannequin'
        
        # Load the input and target images
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            # input_image += self.base_img
            target_image = self.transform(target_image)

        prompt_ids = torch.zeros(1, 1)
        if self.tokenizer:
            inputs = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            prompt_ids = inputs.input_ids.squeeze()

        return {
            'filename': target_path.name,
            'input': input_image,
            'target': target_image,
            'base': self.base_img,
            'prompt': prompt,
            'prompt_ids': prompt_ids,
        }

    def __getitem__(self, idx):
        return self.getoneitem(idx)

    def __getitems__(self, idx):
        return [self.getoneitem(i) for i in idx]


########## Test #########
# train_transforms = transforms.Compose(
#     [
#         transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.RandomCrop(512),
#         transforms.Lambda(lambda x: x),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5]),
#     ]
# )

# testdataset = JoinedDataset(transform=train_transforms, tokenizer=None)
# name = testdataset[10]['filename']
# prompt = testdataset[10]['prompt']
# print(len(testdataset), name, prompt)

# def collate_fn(examples):
#     input_pixel_values = torch.stack([example["input"] for example in examples])
#     input_pixel_values = input_pixel_values.to(memory_format=torch.contiguous_format).float()
#     target_pixel_values = torch.stack([example["target"] for example in examples])
#     target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
#     input_ids = torch.stack([example["prompt_ids"] for example in examples])
#     return {"input_pixel_values": input_pixel_values, "target_pixel_values": target_pixel_values, "input_ids": input_ids}

# train_dataloader = DataLoader(
#     testdataset,
#     shuffle=True,
#     collate_fn=collate_fn,
#     batch_size=16,
# )

# total_samples = 0
# for batch in train_dataloader:
#     total_samples += batch["input_pixel_values"].size(0)

# print('total_samples: ', total_samples)
