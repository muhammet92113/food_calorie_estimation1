import torch

print("PyTorch Version:", torch.__version__)

if torch.cuda.is_available():
    print("GPU bulundu")
    print(torch.cuda.get_device_name(0))
else:
    print("GPU bulunamadı")