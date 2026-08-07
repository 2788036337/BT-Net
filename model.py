import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# -------------权重初始化函数----------------------------------------
def init_weights(*modules):
    """
    初始化神经网络模块的权重
    - Conv2d: 使用Kaiming正态初始化
    - BatchNorm2d: 权重设为1，偏置设为0
    - Linear: 使用Kaiming正态初始化
    """
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

# -----------------Transformer组件-----------------

class Residual(nn.Module):
    """
    残差连接模块：输入 + 模块输出
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    """
    预归一化模块：LayerNorm + 函数
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)  # 层归一化
        self.fn = fn  # 要应用的函数

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    """
    前馈网络：两层线性变换 + 激活 + Dropout
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 扩展维度
            nn.LeakyReLU(),  # 激活
            nn.Dropout(dropout),  # Dropout
            nn.Linear(hidden_dim, dim),  # 收缩回原维度
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """
    多头自注意力机制
    """
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads  # 内部维度
        project_out = not (heads == 1 and dim_head == dim)  # 是否需要输出投影

        self.heads = heads
        self.scale = dim_head ** -0.5  # 缩放因子

        # QKV投影
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # 输出投影
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads  # b:批大小, n:序列长度, h:头数
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # 分成Q,K,V
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # 重排列为多头形式

        # 计算注意力分数
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # 应用掩码（如果有）
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)  # softmax归一化

        # 加权求和
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # 重排列回原形式
        out = self.to_out(out)  # 输出投影
        return out

class Transformer_E(nn.Module):
    """
    Transformer编码器：用于编码输入特征
    包含多层注意力 + 前馈网络
    """
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64*64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 位置编码（已注释）
        # self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        # 构建编码器层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),  # 注意力
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))  # 前馈
            ]))

    def forward(self, x, mask=None):
        # 位置编码（已注释）
        # pos = self.pos_embedding
        # x += pos
        # 通过每一层
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class Transformer_D(nn.Module):
    """
    Transformer解码器：用于解码和生成高频细节
    包含多层双注意力 + 前馈网络
    """
    def __init__(self, dim, depth=2, heads=3, dim_head=16, mlp_dim=48, sp_sz=64*64, num_channels=48, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        # 位置编码（已注释）
        # self.pos_embedding = nn.Parameter(torch.randn(1, sp_sz, num_channels))
        # 构建解码器层
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),  # 第一个注意力
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),  # 第二个注意力
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))  # 前馈
            ]))

    def forward(self, x, mask=None):
        # 位置编码（已注释）
        # pos = self.pos_embedding
        # x += pos
        # 通过每一层
        for attn1, attn2, ff in self.layers:
            x = attn1(x, mask=mask)
            x = attn2(x, mask=mask)
            x = ff(x)
        return x


class SpectralMambaBlock(nn.Module):
    """A lightweight Mamba-like spectral block without external deps.

    It applies gated spectral mixing and a simple state accumulation scan
    along channel order for each spatial location.
    """

    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        # Gated projection on spectral tokens.
        self.in_proj = nn.Linear(channels, channels * 2)

        # Depthwise-like spectral mixing over token order (channel axis as sequence).
        self.mix_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size - 1,
            bias=True,
        )

        # Learnable scan parameters for a simple selective-state accumulation.
        self.scan_a = nn.Parameter(torch.zeros(channels))
        self.scan_b = nn.Parameter(torch.ones(channels))

        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):
        # x: [B, C, H, W] -> process each pixel as a spectral sequence of length C.
        b, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
        seq = self.norm(seq)

        u, v = self.in_proj(seq).chunk(2, dim=-1)

        # Local spectral mixing.
        u_mix = self.mix_conv(u.unsqueeze(1)).squeeze(1)
        u_mix = u_mix[:, :c]

        gated = torch.tanh(u_mix) * torch.sigmoid(v)

        # Cumulative state scan along spectral order.
        a = torch.sigmoid(self.scan_a).view(1, c)
        b_gain = torch.tanh(self.scan_b).view(1, c)
        state = torch.cumsum(gated * a, dim=1)
        y = gated + b_gain * state

        y = self.out_proj(y)
        y = y + seq
        out = y.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        return out


class BTNet(nn.Module):
    """BT-Net: Transformer backbone + Mamba spectral enhancement.

    Pipeline:
    1) Transformer encoder extracts global context on tokens.
    2) Mamba-like spectral block refines spectral dynamics in feature maps.
    3) Transformer decoder reconstructs high-frequency details.
    """

    def __init__(self, num_channel=31, msi_channels=3, num_feature=48,
                 mamba_layers=1, scale_factor=4):
        super().__init__()
        self.num_channel = int(num_channel)
        self.msi_channels = int(msi_channels)
        self.scale_factor = int(scale_factor)
        self.num_feature = num_feature

        self.embedding = nn.Linear(self.num_channel + self.msi_channels, num_feature)
        self.t_e = Transformer_E(num_feature)
        self.spectral_blocks = nn.Sequential(*[
            SpectralMambaBlock(num_feature) for _ in range(mamba_layers)
        ])
        self.t_d = Transformer_D(num_feature)
        self.refine = nn.Sequential(
            nn.Conv2d(num_feature, num_feature, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(num_feature, self.num_channel, 3, 1, 1),
        )
        # Start residual learning from the bicubic baseline.  In particular,
        # do not let a randomly initialized residual dominate the input at the
        # beginning of training.
        nn.init.zeros_(self.refine[-1].weight)
        if self.refine[-1].bias is not None:
            nn.init.zeros_(self.refine[-1].bias)

    def forward(self, HSI, MSI):
        if HSI.shape[1] != self.num_channel:
            raise ValueError(f"Expected {self.num_channel} HSI bands, got {HSI.shape[1]}")
        if MSI.shape[1] != self.msi_channels:
            raise ValueError(f"Expected {self.msi_channels} MSI channels, got {MSI.shape[1]}")
        UP_LRHSI = F.interpolate(HSI, scale_factor=self.scale_factor, mode='bicubic')
        sz = UP_LRHSI.size(2)

        data = torch.cat((UP_LRHSI, MSI), dim=1)
        tokens = rearrange(data, 'B c H W -> B (H W) c', H=sz)
        tokens = self.embedding(tokens)

        code = self.t_e(tokens)

        feat_2d = rearrange(code, 'B (H W) C -> B C H W', H=sz)
        feat_2d = self.spectral_blocks(feat_2d)
        code = rearrange(feat_2d, 'B C H W -> B (H W) C', H=sz)

        highpass = self.t_d(code)
        highpass = rearrange(highpass, 'B (H W) C -> B C H W', H=sz)
        highpass = self.refine(highpass)

        # Keep the training path differentiable.  Clamp only copies used for
        # validation/test metrics and visualisation.
        output = UP_LRHSI + highpass
        return output, UP_LRHSI, highpass


def build_model(model_name="bt-net", num_channel=31, msi_channels=3,
                num_feature=48, mamba_layers=1, scale_factor=4):
    """Build the project's only supported network: BT-Net."""
    name = (model_name or "bt-net").strip().lower()
    if name not in ("bt-net", "btnet", "bt_net"):
        raise ValueError(f"Unsupported model_name: {model_name}. Only 'bt-net' is available.")
    return BTNet(num_channel=num_channel, msi_channels=msi_channels,
                 num_feature=num_feature, mamba_layers=mamba_layers,
                 scale_factor=scale_factor)
