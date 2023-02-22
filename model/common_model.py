import torch
from torch import nn
from torch.nn import functional as F


class MMDL(nn.Module):
    def __init__(self, image_dim, encoders, fusion, head):
        super(MMDL, self).__init__()
        self.encoders = nn.ModuleList(encoders)
        self.fuse = fusion
        self.head = head
        self.attention_block = AttentionBlock(image_dim)

    def forward(self, inputs):
        outs = [self.encoders[0](inputs[0])]

        reduced_image = []
        start_index = 0
        for num in inputs[2]:
            image_feature = self.encoders[1](inputs[1][start_index: start_index + int(num.item())])
            reduced_image.append(self.attention_block(image_feature))
            start_index += int(num.item())
        reduced_image = torch.stack(reduced_image)

        outs.append(reduced_image)
        out = self.fuse(outs)

        return self.head(out)


class MajorityVoting(nn.Module):
    def __init__(self, encoders):
        super(MajorityVoting, self).__init__()
        self.encoders = nn.ModuleList(encoders)

    def forward(self, inputs):
        outs = []
        audio_classes = self.encoders[0](inputs[0])

        start_index = 0
        for (index, num) in enumerate(inputs[2]):
            weight = 1 / (1 + num)
            image_class = self.encoders[1](inputs[1][start_index: start_index + int(num.item())])
            image_class = torch.cat((image_class, torch.unsqueeze(audio_classes[index], 0)), 0)
            result = torch.sum(image_class * weight, dim=0)
            outs.append(result)
        return torch.stack(outs, dim=0)


class MLP(torch.nn.Module):
    """Two layered perceptron."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1, output_each_layer=False):
        """Initialize two-layered perceptron.
        Args:
            indim (int): Input dimension
            hiddim (int): Hidden layer dimension
            outdim (int): Output layer dimension
            dropout (bool, optional): Whether to apply dropout or not. Defaults to False.
            dropoutp (float, optional): Dropout probability. Defaults to 0.1.
            output_each_layer (bool, optional): Whether to return outputs of each layer as a list. Defaults to False.
        """
        super(MLP, self).__init__()
        self.fc = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout_layer = torch.nn.Dropout(dropoutp)
        self.dropout = dropout
        self.output_each_layer = output_each_layer
        self.lklu = nn.LeakyReLU(0.2)

    def forward(self, x):
        """Apply MLP to Input.
        Args:
            x (torch.Tensor): Layer Input
        Returns:
            torch.Tensor: Layer Output
        """
        output = F.relu(self.fc(x))
        if self.dropout:
            output = self.dropout_layer(output)
        output2 = self.fc2(output)
        if self.dropout:
            output2 = self.dropout_layer(output)
        if self.output_each_layer:
            return [0, x, output, self.lklu(output2)]
        return output2


class Identity(nn.Module):
    """Identity Module."""

    def __init__(self):
        """Initialize Identity Module."""
        super().__init__()

    def forward(self, x):
        """Apply Identity to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        return x


class AttentionBlock(nn.Module):
    def __init__(self, input_size, scale=0.8):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.scale = scale

    def forward(self, x):
        x = self.fc1(x)
        weight = self.fc2(self.tanh(x))
        weight = self.softmax(weight * self.scale)

        return torch.sum(x * weight, dim=0)
