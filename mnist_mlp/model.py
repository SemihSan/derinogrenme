from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_sizes=(128, 64)):
        super().__init__()
        if len(hidden_sizes) < 2 or any(h <= 0 for h in hidden_sizes):
            raise ValueError("En az iki pozitif genişlikli hidden layer gerekli.")
        layers = [nn.Flatten()]
        previous = 784
        for width in hidden_sizes:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        # CrossEntropyLoss, softmax işlemini kendi içinde uygular: ham logits verilir.
        layers.append(nn.Linear(previous, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images)
