import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import json
from pathlib import Path


@dataclass
class CellFMConfig:
    n_genes: int = 24078
    enc_dims: int = 1536
    enc_nlayers: int = 2
    enc_num_heads: int = 48
    enc_dropout: float = 0.1
    pad_zero: bool = False
    add_zero: bool = False


class SRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = torch.linalg.vector_norm(x * self.scale, dim=-1, keepdim=True)
        return (x / norm.clamp_min(self.eps)).to(dtype)


class ValueEncoder(nn.Module):
    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.w1 = nn.Linear(1, hidden, bias=False)
        self.w3 = nn.Linear(hidden, hidden, bias=False)
        self.table = nn.Linear(hidden, dim, bias=False)
        self.a = nn.Parameter(torch.zeros(1, 1))
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        # x: (B, L) or masked input (B, L, 2), where [:, :, 0] is unmask flag
        if x.ndim == 3:
            unmask, expr = x.split(1, dim=-1)
            emb = self._encode_value(expr)
            return emb * unmask + self.mask_emb * (1 - unmask), unmask

        expr = x.unsqueeze(-1)
        unmask = torch.ones_like(expr)
        return self._encode_value(expr), unmask

    def _encode_value(self, x):
        # v = F.leaky_relu(self.w1(x))
        v = F.leaky_relu(self.w1(x), negative_slope=0.2)
        v = self.w3(v) + v * self.a
        v = F.softmax(v, dim=-1)
        return self.table(v)


class MHRetention(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_depth=None):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        beta = 1.0 if layer_depth is None else (layer_depth * 8) ** -0.25
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.u_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.inner_norm = SRMSNorm(self.head_dim)

    def _split_heads(self, x):
        b, l, d = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x):
        b, h, l, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, h * d)

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        y = x if y is None else y

        q = F.relu(self._split_heads(self.q_proj(x)))
        k = F.relu(self._split_heads(self.k_proj(y)))
        v = self._split_heads(self.v_proj(y))
        u = F.silu(self._split_heads(self.u_proj(x)))

        if seq_mask is not None:
            q = q * seq_mask
        if attn_mask is not None:
            k = k * attn_mask
        if v_pos is not None:
            v = v * v_pos

        q = q / self.scale
        k = k / self.scale

        kv = k.transpose(-2, -1) @ v
        out = q @ kv
        out = self.inner_norm(out)
        out = out * u

        return self.o_proj(self._merge_heads(out))


class GatedLinearUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.u_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.u_proj(x) * self.v_proj(x))


class RetentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_depth: int, dropout: float = 0.0):
        super().__init__()
        self.alpha = (2 * layer_depth) ** 0.25
        self.attn = MHRetention(dim, num_heads, layer_depth)
        self.ffn = GatedLinearUnit(dim)
        self.dropout = nn.Dropout(dropout)
        # self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-7)
        self.norm2 = nn.LayerNorm(dim, eps=1e-7)

    def forward(self, x, y=None, v_pos=None, attn_mask=None, seq_mask=None):
        out = self.dropout(self.attn(x, y=y, v_pos=v_pos, attn_mask=attn_mask, seq_mask=seq_mask))
        x = self.norm1(x * self.alpha + out)
        out = self.dropout(self.ffn(x))
        x = self.norm2(x * self.alpha + out)
        return x


class CellFM(nn.Module):
    def __init__(self, config: CellFMConfig):
        super().__init__()
        self.config = config
        self.depth = config.enc_nlayers
        self.pad_zero = config.pad_zero
        self.add_zero = config.add_zero

        padded_genes = config.n_genes + 1 + (-(config.n_genes + 1) % 8)
        self.gene_emb = nn.Parameter(torch.empty(padded_genes, config.enc_dims))
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.enc_dims))
        self.zero_emb = nn.Parameter(torch.zeros(1, 1, config.enc_dims))

        self.value_enc = ValueEncoder(config.enc_dims)
        self.encoder = nn.ModuleList([
            RetentionLayer(
                config.enc_dims,
                config.enc_num_heads,
                config.enc_nlayers,
                config.enc_dropout * i / config.enc_nlayers,
            )
            for i in range(config.enc_nlayers)
        ])

        self.norm = SRMSNorm(config.enc_dims)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.gene_emb, gain=0.5)
        nn.init.xavier_normal_(self.cls_token, gain=0.5)
        with torch.no_grad():
            self.gene_emb[0].zero_()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name="CellFM-80M",
        assets_dir="assets",
        map_location="cpu",
    ):
        model_dir = Path(assets_dir) / model_name
    
        config_path = model_dir / "config.json"
        weight_path = model_dir / "model.pt"
    
        if not config_path.exists():
            raise FileNotFoundError(f"Missing {config_path}")
    
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing {weight_path}")
    
        with open(config_path) as f:
            cfg = CellFMConfig(**json.load(f))
    
        model = cls(cfg)
    
        state = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(state, strict=True)
    
        return model

    def encode_tokens(self, expr, gene, zero_idx):
        """
        expr:     (B, L) or (B, L, 2)
        gene:     (B, L), integer gene ids
        zero_idx: (B, L + 1), attention/padding mask including CLS position

        This is graph with CLS, return: [CLS, gene1, gene2, ...]
        """
        # b, l = gene.shape
        b, l = expr.shape[:2]

        # gene_emb = self.gene_emb[gene]
        if gene.ndim == 1:
            gene_emb = self.gene_emb[gene].unsqueeze(0).expand(b, -1, -1)
        else:
            gene_emb = self.gene_emb[gene]
            
        expr_emb, unmask = self.value_enc(expr)
        if not self.pad_zero:
            zero_mask = (1 - zero_idx[:, 1:]).unsqueeze(-1).to(dtype=unmask.dtype)
            zero_unmask = zero_mask * unmask
            expr_emb = zero_unmask * self.zero_emb + (1 - zero_unmask) * expr_emb

        len_scale = (zero_idx.sum(dim=-1, keepdim=True) - 1).rsqrt()
        len_scale = len_scale.detach().view(b, 1, 1, 1)

        x = gene_emb + expr_emb
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)

        if self.pad_zero:
            x = x * zero_idx.view(b, -1, 1)

        mask_pos = torch.cat(
            [torch.ones(b, 1, 1, device=x.device, dtype=unmask.dtype), unmask],
            dim=1,
        ).view(b, 1, -1, 1)

        for i in range(self.depth // 2):
            x = self.encoder[i](x, v_pos=len_scale, attn_mask=mask_pos)

        mask_pos = zero_idx.view(b, 1, -1, 1) if self.pad_zero else None

        for i in range(self.depth // 2, self.depth):
            x = self.encoder[i](x, v_pos=len_scale, attn_mask=mask_pos)

        return x
    
    def encode_genes(self, expr, gene, zero_idx):
        # b, l = gene.shape
        b, l = expr.shape[:2]
    
        # gene_emb = self.gene_emb[gene]
        if gene.ndim == 1:
            gene_emb = self.gene_emb[gene].unsqueeze(0).expand(expr.shape[0], -1, -1)
        else:
            gene_emb = self.gene_emb[gene]
            
        expr_emb, _ = self.value_enc(expr)
        
        len_scale = zero_idx.sum(dim=-1, keepdim=True).rsqrt()
        len_scale = len_scale.detach().view(b, 1, 1, 1)
    
        x = gene_emb + expr_emb
    
        for layer in self.encoder:
            x = layer(x, v_pos=len_scale, attn_mask=None, seq_mask=None)
    
        return x
    
    def encode_cells(self, expr, gene, zero_idx):
        x = self.encode_tokens(expr, gene, zero_idx)
        return x[:, 0]

    @torch.inference_mode()
    def encode(
        self,
        expr,
        gene,
        zero_idx,
        batch_size=64,
        embedding_type="cell",
    ):
        self.eval()

        device = next(self.parameters()).device
        outputs = []

        gene_device = (
            gene.to(device, non_blocking=True)
            if gene.ndim == 1
            else None
        )

        use_amp = device.type == "cuda"

        for start in range(0, expr.shape[0], batch_size):
            end = min(start + batch_size, expr.shape[0])

            expr_b = expr[start:end].to(device, non_blocking=True)

            if gene.ndim == 1:
                gene_b = gene_device
            else:
                gene_b = gene[start:end].to(device, non_blocking=True)

            zero_b = zero_idx[start:end].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                if embedding_type == "cell":
                    emb = self.encode_cells(expr_b, gene_b, zero_b)
                elif embedding_type == "gene":
                    emb = self.encode_genes(expr_b, gene_b, zero_b)
                else:
                    raise ValueError(
                        "embedding_type must be 'cell' or 'gene'."
                    )

            outputs.append(emb.float().cpu())

        return torch.cat(outputs, dim=0)

    def forward(self, expr, gene, zero_idx):
        gene_emb = self.encode_genes(expr, gene, zero_idx)
        return gene_emb, None