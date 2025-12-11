import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch built with cuda:", torch.version.cuda)
