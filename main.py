import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_f, out_f, r=4):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)

    def forward(self, x):
        return self.linear(x) + self.B(self.A(x))
