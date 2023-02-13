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

        image_feature = []
        for input in inputs[1].permute(1, 0, 2, 3, 4):
            image_feature.append(self.encoders[1](input))
        image_feature = torch.stack(image_feature).permute(1,0,2)
        reduced_image = []
        for image in image_feature:
            reduced_image.append(self.attention_block(image))
        reduced_image = torch.stack(reduced_image)

        outs.append(reduced_image)
        out = self.fuse(outs)

        return self.head(out)


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
    def __init__(self, input_size, scale=0.3):
        super(AttentionBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size, bias=False)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.scale = scale

    def forward(self, x):
        x = self.fc1(x)
        weight = self.fc2(self.tanh(x))
        weight = self.softmax(weight * self.scale)

        return torch.sum(x * weight, dim=0)
