from torchvision.datasets import Food101

Food101(
    root="../dataset",
    split="train",
    download=True
)

Food101(
    root="../dataset",
    split="test",
    download=True
)

print("Food-101 indirildi.")