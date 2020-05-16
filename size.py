import os
import torch

model_dir = './16_bit_model.pkl'

print(f"original cost: {os.stat(model_dir).st_size} bytes.")
