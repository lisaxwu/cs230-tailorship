from img_pairs_dataset import JoinedDataset

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from custom_unet import UNet
import argparse
from torchvision import transforms
import time
import torch.optim as optim
from torch.optim import lr_scheduler

def get_preprocess_transform(resolution):
	return transforms.Compose(
		[
			transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
			transforms.CenterCrop(resolution),
			transforms.ToTensor(),
		]
	)

def train(args):
	# 1. Dataset
	preprocess_transform = get_preprocess_transform(args.resolution)
	print("Input + Target as Input!!")
	train_dataset = JoinedDataset(transform=preprocess_transform, tokenizer=None)

	def collate_fn(examples):
		input_pixel_values = torch.stack([example["input"] for example in examples])
		input_pixel_values = input_pixel_values.to(memory_format=torch.contiguous_format).float()
		target_pixel_values = torch.stack([example["target"] for example in examples])
		target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
		input_ids = torch.stack([example["prompt_ids"] for example in examples])
		return {"input_pixel_values": input_pixel_values, "target_pixel_values": target_pixel_values, "input_ids": input_ids}

	train_dataloader = torch.utils.data.DataLoader(
		train_dataset,
		shuffle=True,
		collate_fn=collate_fn,
		batch_size=args.batch_size,
	)

	# 2. Train
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = UNet(3)
	model = model.to(device)
	model.train()
	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

	train_loss = 0.0
	since = time.time()
	for epoch in range(0, args.num_train_epochs):
		for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}", leave=True):
			input_img = batch["input_pixel_values"].to(dtype=torch.float32).to(device)
			target_img = batch["target_pixel_values"].to(dtype=torch.float32).to(device)
			input_img += target_img

			optimizer.zero_grad()
			outputs = model(input_img)
			loss = F.mse_loss(outputs.float(), target_img.float(), reduction="mean")
			loss.backward()
			optimizer.step()
			train_loss += loss

		print('Epoch %d finished! Avg train loss over epochs = %f' % (epoch, train_loss / (epoch + 1)))
		time_elapsed = time.time() - since
		print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
		scheduler.step()

	torch.save(model.state_dict(), args.model_path)

def parse_args():
	parser = argparse.ArgumentParser(description="Simple example of a training script.")
	parser.add_argument(
		"--batch_size", type=int, default=16, help="Batch size."
	)
	parser.add_argument(
		"--num_train_epochs", type=int, default=20, help="Train Epochs."
	)
	parser.add_argument(
        "--model_path",
        type=str,
        default="custom-unet/unet.pt",
        help="The output directory where the model state is save / loaded.",
    )
	parser.add_argument(
		"--resolution",
		type=int,
		default=512,
		help=(
		    "The resolution for input images, all the images in the train/validation dataset will be resized to this"
		    " resolution"
		),
	)
	parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
	args = parser.parse_args()
	return args


if __name__ == "__main__":
	args = parse_args()
	train(args)
