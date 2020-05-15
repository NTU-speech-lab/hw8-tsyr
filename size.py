import os
import torch

model_dir = './student_model.bin'

print(f"original cost: {os.stat(model_dir).st_size} bytes.")
