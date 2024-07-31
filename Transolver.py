class Transolver(nn.Module):
    def __init__(self, dim, H, W, num_blocks):
        super(Transolver, self).__init__()
        self.dim = dim
        self.H = H
        self.W = W
        self.num_blocks = num_blocks
        self.physics_attentions = nn.ModuleList([Physics_Attention_Structured_Mesh_2D(dim, H=H, W=W) for _ in range(num_blocks)])
        self.layer_norms_hats = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_blocks)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_blocks)])
        self.feedforwards = nn.ModuleList([FeedForward(H*W) for _ in range(num_blocks)])
        self.initial = Linear(H*W, H*W)
        self.final = Linear(H*W, H*W, bias=False)

    def forward(self, x):
        x = self.initial(x)
        for i, physics_attention in enumerate(self.physics_attentions):
            x_hat = x + physics_attention(self.layer_norms_hats[i](x))
            x = x_hat + self.feedforwards[i](self.layer_norms[i](x_hat))
        return self.final(x)


class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.linear(x).reshape(x.shape[0], -1, 1)


class FeedForward(nn.Module):
    def __init__(self, in_dim):
        super(FeedForward, self).__init__()
        self.in_dim = in_dim
        self.linears = nn.ModuleList([nn.Linear(in_dim, 256), nn.Linear(256, 128), nn.Linear(128, 256), nn.Linear(256, in_dim)])

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for l in self.linears:
            x = torch.tanh(l(x))
        return x.reshape(x.shape[0], -1, 1)

