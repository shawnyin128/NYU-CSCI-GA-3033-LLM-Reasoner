import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print("input to model:", x.dtype)

        x = self.fc1(x)
        print("after fc1:", x.dtype)

        x = self.relu(x)
        print("after relu:", x.dtype)

        x = self.ln(x)
        print("after ln:", x.dtype)

        x = self.fc2(x)
        print("after fc2:", x.dtype)

        return x


device = "cuda"
model = ToyModel(in_features=4, out_features=2).to(device)

print("fc1 weight dtype:", model.fc1.weight.dtype)
print("ln weight dtype:", model.ln.weight.dtype)
print("fc2 weight dtype:", model.fc2.weight.dtype)

x = torch.randn(3, 4, device=device, dtype=torch.float16)
print("input x dtype before forward:", x.dtype)

with torch.autocast(device_type="cuda", dtype=torch.float16):
    y = model(x)

print("output y dtype:", y.dtype)