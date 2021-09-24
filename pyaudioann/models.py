class LSTM(nn.Module):
    """
    Standard LSTM class
    """
    def __init__(self, input_size, hidden_size, num_layers, seq_len=1,
                 dev='cpu'):
        """
        Constructor of LSTM class
        Args:
            input_size (int):   Input size of LSTM
            hidden_size (int):  Number of units per hidden layer of LSTM
            num_layers (int):   Number of LSTM layers
            seq_len (int):      Sequence length
            dev (torch device): CPU or GPU device
        Returns:
        """
        super(LSTM, self).__init__()
        self.n_layers = num_layers
        self.hidden_dim = hidden_size
        self.dev = dev

        # Define LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        # Define a FC layer
        self.fc = nn.Linear(hidden_size * seq_len, input_size)

        # Initialize FC layer weights
        nn.init.xavier_uniform_(self.fc.weight)

        # Initialize LSTM's weights and biases
        for layer in self.lstm._all_weights:
            for p in layer:
                if 'weight' in p:
                    nn.init.xavier_uniform_(self.lstm.__getattr__(p).data)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = self.lstm.__getattr__(name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
        # Flag is used for initializing the states of LSTM (h0, c0)
        self.flag = 0

    def forward(self, x):
        """
        Forward method of LSTM class.
        Args:
            x (tensor): Input tensor (batch_size, sequence_lengh, features_dim)
        Returns:
            Tensor of size (batch_size, sequence_lengh, features_dim)
        """
        batch_size = x.shape[0]
        # Initialize LSTM states
        if self.flag == 0:
            self.h0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.c0 = nn.Parameter(randn(self.n_layers*1, batch_size,
                                         self.hidden_dim),
                                   requires_grad=True).to(self.dev)
            self.flag = 1
        out, (self.h, self.c) = self.lstm(x, (self.h0, self.c0))
        m, n = out.shape[1], out.shape[2]
        out = out.reshape(-1, m*n)
        out = self.fc(out)
        return out
