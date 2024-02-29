from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.Resize((704,608)),
    transforms.ToTensor(),
])