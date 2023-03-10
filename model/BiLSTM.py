import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, dropout=0.1, linear=False, out_size = 6):
        super(BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_1, batch_first=True, bidirectional=True)
        self.hidden1 = nn.Linear(2 * hidden_1, hidden_1)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(hidden_2, out_size)
        self.dropout = nn.Dropout(dropout)

        self.linear = linear

    def forward(self, x):
        emb, _ = self.bilstm(x)
        emb = emb[:, -1]
        emb = self.hidden1(emb)
        emb = self.relu2(emb)

        emb = self.hidden2(emb)
        emb = self.relu2(emb)

        emb = self.dropout(emb)
        if self.linear:
            emb = self.fc(emb)
        return emb