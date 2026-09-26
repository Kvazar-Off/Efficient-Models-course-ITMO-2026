import torch
import torch.nn as nn


def build_model(num_classes=100):
    layers = [
        nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2, bias=False),
        nn.ReLU(inplace=True),

        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),

        nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0, bias=False),
        nn.ReLU(inplace=True),
    ]
    features = nn.Sequential(*layers)

    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),   
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )
    return nn.Sequential(features, head)


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_model().cuda().eval()
    x = torch.randn(2, 3, 64, 64, device="cuda")
    with torch.inference_mode():
        y = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print("выход:", tuple(y.shape))
    print("параметров:", n_params)
