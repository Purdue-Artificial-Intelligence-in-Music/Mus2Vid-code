import torch
import math

class CNN_model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, device="cuda:0" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.input_size = input_size
        self.length_input = self.input_size[0]
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,13), padding='same', device=self.device, dtype=torch.double), 
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AvgPool2d(kernel_size=(1,3)),
            torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,3), padding='same', device=self.device, dtype=torch.double),
            torch.nn.Tanh(),
            torch.nn.BatchNorm2d(1),
            torch.nn.AvgPool2d(kernel_size=(1,3)), 
        )

        self.MLP_layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=int(math.floor(self.input_size[1] / 9)), out_features=self.hidden_size, device=self.device, dtype=torch.double),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=self.hidden_size, out_features=self.out_size[1], device=self.device, dtype=torch.double)
        )

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        conv_out = self.conv_layers(x).squeeze_()
        out = torch.reshape(conv_out, (conv_out.shape[0] * conv_out.shape[1], -1))
        out = self.MLP_layers(out)
        out = torch.reshape(out, (conv_out.shape[0] , conv_out.shape[1], -1))

        return out
