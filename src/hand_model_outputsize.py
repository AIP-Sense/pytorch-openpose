import torch
from tqdm import tqdm
import json

from src.model import handpose_model
from src import util

model = handpose_model()
device = util.get_torch_device()
model = model.to(device)

size = {}
for i in tqdm(range(10, 1000)):
    data = torch.randn(1, 3, i, i).to(device)
    size[i] = model(data).size(2)

with open('hand_model_output_size.json') as f:
    json.dump(size, f)
