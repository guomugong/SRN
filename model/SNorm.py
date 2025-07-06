import torch.nn as nn

# style randomization module
class SNorm(nn.Module):
    def __init__(self, num_features=1):
        super().__init__()
        self.num_features = num_features
        self.gamma = nn.Sequential(
           nn.AdaptiveAvgPool2d((1, 1)), 
           nn.Flatten(),
           nn.Linear(self.num_features, self.num_features*2),
           nn.ReLU(),
           nn.Linear(self.num_features*2, num_features))

        self.beta = nn.Sequential(
           nn.AdaptiveAvgPool2d((1, 1)), 
           nn.Flatten(),
           nn.Linear(self.num_features, self.num_features*2),
           nn.ReLU(),
           nn.Linear(self.num_features*2, num_features))

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, C, -1)

        # normalization
        mean = x.mean(dim=2, keepdim=True)  # [N, C, 1]
        std = x.std(dim=2, keepdim=True)    # [N, C, 1]
        norm_feat = (x - mean) / (std + 1e-5)
        norm_feat = norm_feat.view(N, C, H, W)

        # dynamic affine transformation
        gamma = self.gamma(norm_feat)
        beta = self.beta(norm_feat)
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # 转换为 [batch_size, channels, 1, 1]
        beta = beta.unsqueeze(2).unsqueeze(3)
        aff_feat = gamma * norm_feat + beta

        return aff_feat 
