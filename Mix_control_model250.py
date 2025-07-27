import torch
import torch.nn as nn

class EEG_MutiCovnet(nn.Module):
    def __init__(self): #输入 (32, 250)的EMG数据，输出xyz的坐标
        super().__init__()

        # eegnet前特征提取
        self.eeg_first_cov = nn.Sequential(
            # 特征提取
            nn.Conv2d(1, 16, kernel_size=(1,63),
                      stride=1, padding=(0,31),bias=False),  # 输出维度 B * 16 * 32 * T
            nn.BatchNorm2d(16), #批量归一化 tensor处理(1,16,32,1024) 让每一层均值接近零，方差接近1
        )
        # 到此 16 * 32 * T

        # 多分支卷积核卷积，分散的学习每个通道的特征
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
        # 4* (32, 32, 1000)

        # 将四个分支的输出通道数拼接，32*4=128
        self.conv_fusion = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(128)
        # (32, 4*32, 1000)

        # 时间轴池化，假设不降采样，或你可以加AvgPool2d降采样
        self.pool=nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # (B,128,32通道,125时间)


    def forward(self, x):  # x: (B, 250, 32)
        # x = x.unsqueeze(0) #添加一个batch维度 (B, 32, 250)
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


class EMG_MutiCovnet(nn.Module):
    def __init__(self): #输入 (5, 250)的EMG数据，输出xyz的坐标
        super().__init__()

        # eegnet前特征提取
        self.eeg_first_cov = nn.Sequential(
            # 特征提取
            nn.Conv2d(1, 16, kernel_size=(1,63),
                      stride=1, padding=(0,31),bias=False),  # 输出维度 B * 16 * 5 * T
            nn.BatchNorm2d(16), #批量归一化 tensor处理(1,16,5,1024) 让每一层均值接近零，方差接近1
        )
        # 到此 16 * 5 * T

        # 多分支卷积核卷积，分散的学习每个通道的特征
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
        # 4* (32, 5, 250)

        # 将四个分支的输出通道数拼接，32*4=128
        self.conv_fusion = nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.bn_fusion = nn.BatchNorm2d(128)
        # (4*32, 32, 250)

        # 时间轴池化，假设不降采样，或你可以加AvgPool2d降采样
        self.pool=nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))  # (B,128,5通道,125时间)


    def forward(self, x):  # x: (B, 250, 5)
        # x = x.unsqueeze(0) #添加一个batch维度 (B, 5, 250)
        x = x.permute(0, 1, 2).unsqueeze(1)  # (B, 1, 5, 250)

        x = self.eeg_first_cov(x)

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1) #(B, 128, 5, 250)
        x = self.conv_fusion(x)

        x = self.pool(x)
        x = x.permute(0, 2, 1, 3)  # (B, F, C, T) -> (B, C, F, T) (B, 5, 128, 125)

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



class Sensor_SE_Block(nn.Module):
    def __init__(self, num_sensors=5, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_sensors, num_sensors // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_sensors // reduction, num_sensors, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: (B, C, F, T)
        B, C, F, T = x.size()

        y = self.avg_pool(x).view(B, C)  # shape: (B, C, 1, 1)
        y = self.fc(y).view(B, C, 1, 1)

        return x * y.expand_as(x)



class Embedding(nn.Module):
    def __init__(self, total_channels=37, in_features=128, d_model=64):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, d_model, kernel_size=(in_features, 1)),  # 融合 EEG + EMG
            nn.ReLU()
        )

    def forward(self, eeg_x, emg_x):
        """
        eeg_x: (B, C_eeg, F, T)
        emg_x: (B, C_emg, F, T)
        return: (B, T, d_model)
        """
        x = torch.cat([eeg_x, emg_x], dim=1)  # (B, C_eeg + C_emg, F, T)

        x = self.project(x)                  # (B, d_model, 1, T)
        x = x.squeeze(2)                     # (B, d_model, T)
        x = x.permute(0, 2, 1)               # (B, T, d_model)

        return x



class ModalityFusion(nn.Module):
    def __init__(self, dropout_eeg=0.3, dropout_emg=0.1):
        super().__init__()

        self.dropout_eeg = nn.Dropout(dropout_eeg)
        self.dropout_emg = nn.Dropout(dropout_emg)

        # 可学习的模态权重参数（初始为1.0）
        self.eeg_weight = nn.Parameter(torch.tensor(1.0))
        self.emg_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg, emg):
        eeg = self.dropout_eeg(eeg)
        emg = self.dropout_emg(emg)

        # 归一化模态权重（Softmax）
        weights = torch.stack([self.eeg_weight, self.emg_weight], dim=0)
        weights = torch.softmax(weights, dim=0)
        eeg_w, emg_w = weights[0], weights[1]


        eeg, emg = eeg_w * eeg , emg_w * emg

        return eeg, emg



class Muti_Attention(nn.Module):
    def __init__(self,
                 dim, #输入的token维度
                 num_heads =8, #注意力的头数
                 qkv_bias = False, #生成QKV时是否添加偏置
                 qk_scale=None, #用于缩放QK的系数，如果为None，则使用1/sqrt(embed_dim_pre_head)
                 atte_drop_ration=0.1, #z\注意力分数的dropout的比率，防止过拟合
                 proj_drop_ration=0.1 #最终投影曾的dropout的比例
                 ):
        super().__init__()

        self.num_heads=num_heads #注意力头数
        head_dim = dim//num_heads #每个注意力的维度
        self.scale = qk_scale or head_dim ** -0.5 #qk的缩放因子
        self.qkv = nn.Linear(dim,dim*3, bias=qkv_bias) #通过全链接层生成QKV，为了并行计算，提高计算效率，参数更少
        self.att_drop=nn.Dropout(atte_drop_ration)
        self.proj_drop=nn.Dropout(proj_drop_ration)
        #将每个head得到的输出进行concact拼接，然后通过线性变换映射回原本的嵌入维度
        self.proj=nn.Linear(dim,dim)

    def forward(self,x):
        B,N,C = x.shape # batch ,num_patchs+1, embed_dim 这个1为clstoken
        #B N 3*C -> B N 3 num_heads C//self.num_heads
        #B N 3 num_heads C//self.num_heads -> 3, B, num_heads,N,C//self_num_heads
        qkv=self.qkv(x).reshape(B,N,3,self.num_heads,C//self.num_heads).permute(2,0,3,1,4) #方便我们之后做运算
        #用切片拿到QKV，形状 B, num_heads, N, C//self.num_heads
        q,k,v = qkv[0],qkv[1],qkv[2]
        #计算qk的点积， 并进行缩放 得到注意力分数
        #Q :[B, num_heads, N, C//self.num_heads]
        #k.transpose(-2,-1) K:[B, num_heads, N, C//self.num_heads] -> [B, num_heads, C//self.numheads, N]
        attn = (q @ k.transpose(-2,-1))*self.scale #[B, num_heads, N, N]
        attn = attn.softmax(dim=-1) #对每行进行处理 使得每行的和为1
        #注意力权重对v进行加权求和
        #attn @v：B, num_heads, N, C//self.num_heads
        #transpose: B,N. self.num_heads,C//self.num_heads
        #reshape: B,N,C,将最后两个维度信息拼接，合并多个头输出，回到总的嵌入维度
        x=(attn @v).transpose(1,2).reshape(B,N,C)
        #通过线性变换映射回原本的嵌入dim
        x=self.proj(x)
        x=self.proj_drop(x) #防止过拟合

        return x

class Connected_layer(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出 6D 坐标
        )

    def forward(self, x):  # x.shape = (B, N, C)
        x = x.mean(dim=1)  # 或者用 x[:, 0, :] 取第一个token
        out = self.mlp(x)  # 输出 shape: (B, 6)
        return out


class EEGTransformerModel(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.eeg_feature_extractor = EEG_MutiCovnet()
        self.emg_feature_extractor = EMG_MutiCovnet()

        self.EEG_SE_block = SE_block()
        self.EMG_SE_block = Sensor_SE_Block()

        self.fusion = ModalityFusion(dropout_eeg=0.3, dropout_emg=0.1)

        self.embedding = Embedding()
        self.attention = Muti_Attention(dim=64)
        self.coord_regressor = Connected_layer(output_dim=output_dim)

    def forward(self, x, y):  # x: (B, 250, 32) y: (B, 250, 5)
        x = self.eeg_feature_extractor(x)    # (B, 32, 125)
        y = self.emg_feature_extractor(y)    # (B, 5, 125)

        x = self.EEG_SE_block(x)
        y = self.EMG_SE_block(y)

        x,y=self.fusion(x,y)

        x = self.embedding(x,y)            # (B, N, 512)
        x = self.attention(x)            # (B, N, 512)
        coords = self.coord_regressor(x) # (B, 6)

        return coords


#
# # # 模型实例
# model = EEGTransformerModel()
#
# # 输入：batch_size = 8 batch，32通道，250时间点
# x = torch.randn(8, 32, 250)
# y = torch.randn(8,5,250)
# out = model(x,y)
#
# print(out.shape)  # (8, 6)








