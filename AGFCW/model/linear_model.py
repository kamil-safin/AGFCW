import torch
import torch.nn as nn


class LogisticRegression(nn.Module):

    def __init__(self, dim, n_classes=2, bias=True, criterion='cross_entropy') -> None:
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(dim, n_classes, bias=bias)
        self.n_dim = dim
        self.n_classes = n_classes
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.params = list(self.parameters())
        self._W_size = self.n_classes*self.n_dim
        if criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.linear(X)

    def get_param(self, flatten_index):
        param_idx = int(flatten_index >= self._W_size)
        coord_idx = flatten_index % self._W_size
        return self.params[param_idx].data.flatten()[coord_idx]

    def get_loss(self, X, y):
        with torch.no_grad():
            out = self.forward(torch.FloatTensor(X))
            loss = self.criterion(out, torch.LongTensor(y))
        return loss.item()
