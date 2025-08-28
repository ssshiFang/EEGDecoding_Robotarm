import torch
import torch.nn as nn

class MutiCovnet(nn.Module):
    def __init__(self):  # Input: EEG data (32, 250), Output: xyz coordinates
        super().__init__()

        # Initial feature extraction (EEGNet style)
        self.eeg_first_cov = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 63),
                      stride=1, padding=(0, 31), bias=False),  # Output: B x 16 x 32 x T
            nn.BatchNorm2d(16),  # Batch normalization to make mean~0, std~1
        )
        # Output shape: 16 x 32 x T

        # Multi-branch convolution to learn channel features separately
        self.branch1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 11), padding=(0, 5), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU()
        )
        # 4 branches output shapes: 32 x 32 x 250 each

        # Concatenate the outputs of four branches along channel dimension (32*4=128)
        self.conv_fusion = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(128)
        # Output shape: B x 128 x 32 x 250

        # Temporal pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # Output: B x 128 x 32 x 125



    def forward(self, x):  # x: (B, 250, 32)
        # x = x.unsqueeze(0) #add demision (B, 32, 250)
        x = x.permute(0, 1, 2).unsqueeze(1)  # (B,1,32,250)

        x = self.eeg_first_cov(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1) #(B, 128, 32, 250)
        x = self.conv_fusion(x)

        x = self.pool(x)
        x = x.permute(0, 2, 1, 3)  # (B, F, C, T) -> (B, C, F, T) (B, 32, 128, 125)

        return x


class SE_block(nn.Module):
    def __init__(self, channels=32, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出 (B, C, 1, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, F, T = x.size()
        y = self.avg_pool(x).view(B, C)  # -> (B, C)
        y = self.fc(y).view(B, C, 1, 1)  # -> (B, C, 1, 1)

        return x * y.expand_as(x)


class Embedding(nn.Module):
    def __init__(self, in_channels=32, in_features=128, d_model=256):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=(in_features, 1)),  # 1x1 conv across features
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, C=32, F=128, T=125)
        x = self.project(x)  # -> (B, d_model, 1, T)
        x = x.squeeze(2)  # -> (B, d_model, T)
        x = x.permute(0, 2, 1)  # -> (B, T, d_model)

        return x



class Muti_Attention(nn.Module):
    def __init__(self,
                 dim,  # input token dimension
                 num_heads=8,  # number of attention heads
                 qkv_bias=False,  # add bias when generating QKV
                 qk_scale=None,  # scale factor for QK; if None, use 1/sqrt(head_dim)
                 atte_drop_ration=0.1,  # dropout rate for attention scores
                 proj_drop_ration=0.1  # dropout rate for output projection
                 ):
        super().__init__()

        self.num_heads = num_heads  # number of attention heads
        head_dim = dim // num_heads  # dimension per head
        self.scale = qk_scale or head_dim ** -0.5  # scaling factor for QK
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # generate Q, K, V with one linear layer
        self.att_drop = nn.Dropout(atte_drop_ration)  # dropout for attention
        self.proj_drop = nn.Dropout(proj_drop_ration)  # dropout for output
        self.proj = nn.Linear(dim, dim)  # project concatenated heads back to original dim

    def forward(self, x):
        B, N, C = x.shape  # batch size, number of tokens, embedding dimension

        # reshape and permute to prepare Q, K, V
        # B, N, 3*C -> B, N, 3, num_heads, C//num_heads -> 3, B, num_heads, N, C//num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # split Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]

        # compute scaled dot-product attention
        # Q: [B, num_heads, N, head_dim]
        # K: [B, num_heads, head_dim, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)  # row-wise softmax

        # apply attention to V
        # attn @ V: [B, num_heads, N, head_dim]
        # transpose and reshape to combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # project back to original embedding dimension
        x = self.proj(x)
        x = self.proj_drop(x)  # dropout to prevent overfitting

        return x


class Connected_layer(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # output 6D
        )

    def forward(self, x):  # x.shape = (B, N, C)
        x = x.mean(dim=1)
        out = self.mlp(x)  # output shape: (B, 6)
        return out


class EEGTransformerModel(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.feature_extractor = MutiCovnet()
        self.SE_block= SE_block()
        self.embedding = Embedding()
        self.attention = Muti_Attention(dim=256)
        self.coord_regressor = Connected_layer(output_dim=6)

    def forward(self, x):  # x: (B, 1024, 32)
        x = self.feature_extractor(x)    # (B, N, 128)
        x = self.SE_block(x)
        x = self.embedding(x)            # (B, N, 512)
        x = self.attention(x)            # (B, N, 512)
        coords = self.coord_regressor(x) # (B, 6)

        return coords



# # model object
# model = EEGTransformerModel()
#
# # input:batch_size = 8 batch，32 channel，250 sampled points
# x = torch.randn(8, 32, 250)
# out = model(x)
#
# print(out.shape)  # (8, 6)








