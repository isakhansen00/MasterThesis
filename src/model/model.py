import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- building blocks ----------------- #

class GLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, d * 2)
    
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return a * torch.sigmoid(b)
    

class GateAddNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gate = GLU(d)
        self.norm = nn.LayerNorm(d)
    
    def forward(self, x, skip):
        return self.norm(self.gate(x) + skip)


class GRN(nn.Module):
    """
    Gated Residual Network (TFT paper).
    d_in  -> d_hidden -> d_out, with optional context.
    """
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: Optional[int] = None,
        ctx_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_out = d_out or d_in

        self.w2 = nn.Linear(d_in, d_hidden)
        self.w3 = nn.Linear(ctx_dim, d_hidden, bias=False) if ctx_dim is not None else None
        self.elu = nn.ELU()
        self.w1 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)

        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.gate_add_norm = GateAddNorm(d_out)

    def forward(self, x, ctx: Optional[torch.Tensor] = None):
        """
        x: (..., d_in)
        ctx: (..., ctx_dim) broadcast to x's leading dims
        """
        y = self.w2(x)
        if self.w3 is not None and ctx is not None:
            while ctx.dim() < y.dim():
                ctx = ctx.unsqueeze(1)
            y = y + self.w3(ctx)
        y = self.elu(y)
        y = self.dropout(self.w1(y))
        return self.gate_add_norm(y, self.skip(x))
    

class ScaledDotAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        query: (B,T_q,d)
        key:   (B,T_k,d)
        value: (B,T_k,d_v)
        mask:  (B,T_q,T_k) with 1=keep, 0=mask
        """
        dk = query.size(-1)
        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.bmm(w, value)
        return out, w


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.n = num_heads
        self.d_k = d_model // num_heads

        self.v_proj = nn.Linear(d_model, self.d_k, bias=False)
        self.q_proj = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(num_heads)])
        self.attn = ScaledDotAttention(dropout)
        self.out = nn.Linear(self.d_k, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        q,k: (B,T,d_model)
        v:   (B,T,d_model)
        attn_mask: (B,T_q,T_k)
        """
        V = self.v_proj(v)
        heads, weights = [], []
        for i in range(self.n):
            Qi = self.q_proj[i](q)
            Ki = self.k_proj[i](k)
            out, w = self.attn(Qi, Ki, V, attn_mask)
            heads.append(out)
            weights.append(w)

        H = torch.stack(heads, dim=2)   # (B,T_q,num_heads,d_k)
        W = torch.stack(weights, dim=2) # (B,T_q,num_heads,T_k)
        H = H.mean(dim=2)               # simple averaging over heads
        out = self.out(H)
        out = self.drop(out)
        return out, W
    

class VariableSelectionNetwork(nn.Module):
    """
    VSN as in TFT:
      - Each variable i: GRN_i(Linear_i(x_i))
      - Flatten all -> GRN_flat -> V logits
      - Softmax over variables -> weighted sum of transformed vars.
    """

    def __init__(
        self,
        var_dims: List[int],      # input dims per variable
        d_model: int,
        ctx_dim: Optional[int],
        dropout: float = 0.1,
    ):
        super().__init__()
        self.V = len(var_dims)
        self.proj = nn.ModuleList([nn.Linear(dv, d_model) for dv in var_dims])
        self.var_grn = nn.ModuleList([
            GRN(d_model, d_model, d_model, ctx_dim, dropout) for _ in var_dims
        ])
        self.flatten_grn = GRN(self.V * d_model, d_model, self.V, ctx_dim, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        xs: List[torch.Tensor],
        ctx: Optional[torch.Tensor],
    ):
        """
        xs: list of tensors, each shape (B,T,d_i)
        ctx: (B,d_ctx) or None
        returns:
          y:     (B,T,d_model)  - weighted sum of transformed variables
          alpha: (B,T,V)        - attention over variables
          H:     (B,T,V,d_model) transformed variables
        """
        Hs = []
        for i, x in enumerate(xs):
            h = self.proj[i](x)
            h = self.var_grn[i](h, ctx)
            Hs.append(h)
        H = torch.stack(Hs, dim=2)  # (B,T,V,d_model)

        flat = H.flatten(start_dim=2)            # (B,T,V*d_model)
        scores = self.flatten_grn(flat, ctx)     # (B,T,V)
        alpha = self.softmax(scores)             # (B,T,V)
        y = torch.sum(alpha.unsqueeze(-1) * H, dim=2)  # (B,T,d_model)
        return y, alpha, H


# ----------------- TFTClassifier with static VSN ----------------- #

class TFTClassifier(nn.Module):
    def __init__(
        self,
        enc_var_dims: Dict[str, int],
        dec_var_dims: Dict[str, int],
        static_cont_dim: int = 0,
        static_cat_cardinalities: Optional[List[int]] = None,
        static_cat_emb_dim: int = 16,
        d_model: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.1,
        num_heads: int = 4,
        num_classes: int = 20,
        pred_len: int = 1,
        eta_quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9),
    ):
        """
        enc_var_dims: e.g. {"lat":1,"lon":1,"sog":1,"cog":1}
        dec_var_dims: e.g. {"dec_time":1} or richer future-known features
        static_cont_dim: number of static continuous features (e.g. 5)
        static_cat_cardinalities: list of cardinalities for each static cat var
        """
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        # ----- static inputs -----
        self.static_cont_dim = static_cont_dim
        self.static_cat_embs = None
        self.n_static_cat = 0
        self.static_cat_emb_dim = static_cat_emb_dim

        static_dim_total = 0

        if static_cat_cardinalities:
            self.n_static_cat = len(static_cat_cardinalities)
            self.static_cat_embs = nn.ModuleList(
                [nn.Embedding(c, static_cat_emb_dim) for c in static_cat_cardinalities]
            )
            static_dim_total += self.n_static_cat * static_cat_emb_dim

        if static_cont_dim > 0:
            static_dim_total += static_cont_dim

        # For static VSN, each static continuous variable is 1-dim,
        # each static categorical variable has emb_dim
        static_var_dims: List[int] = []
        static_var_dims.extend([1] * static_cont_dim)  # each cont is scalar
        static_var_dims.extend([static_cat_emb_dim] * self.n_static_cat)

        if len(static_var_dims) == 0:
            # no static features: use a single dummy variable
            static_var_dims = [1]

        self.vsn_static = VariableSelectionNetwork(
            var_dims=static_var_dims,
            d_model=d_model,
            ctx_dim=None,           # no external context for static VSN
            dropout=dropout,
        )

        # after static VSN, we get a single static context vector s (d_model)
        # from this we derive the three TFT static contexts:
        self.c_selection = GRN(d_model, d_model, d_model, dropout=dropout)
        self.c_enrich    = GRN(d_model, d_model, d_model, dropout=dropout)
        self.c_context   = GRN(d_model, d_model, d_model, dropout=dropout)

        # ----- encoder / decoder variable selection -----
        self.enc_names = list(enc_var_dims.keys())
        self.dec_names = list(dec_var_dims.keys())

        self.vsn_enc = VariableSelectionNetwork(
            [enc_var_dims[k] for k in self.enc_names],
            d_model,
            ctx_dim=d_model,
            dropout=dropout,
        )
        self.vsn_dec = VariableSelectionNetwork(
            [dec_var_dims[k] for k in self.dec_names],
            d_model,
            ctx_dim=d_model,
            dropout=dropout,
        )

        # ----- LSTMs -----
        self.enc_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dec_lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ----- post-LSTM + attention block -----
        self.post_lstm_gating   = GRN(d_model, d_model, d_model, ctx_dim=d_model, dropout=dropout)
        self.static_enrichment  = GRN(d_model, d_model, d_model, ctx_dim=d_model, dropout=dropout)
        self.attn               = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_gate_norm     = GateAddNorm(d_model)
        self.pos_wise_ffn       = GRN(d_model, d_model, d_model, dropout=dropout)

        # ----- heads -----
        self.classifier = nn.Linear(d_model, num_classes)

        self.eta_quantiles = eta_quantiles
        self.eta_head = nn.Linear(d_model, len(eta_quantiles)) if len(eta_quantiles) > 0 else None

    # ------------- helper masks ------------- #

    def _make_causal_mask(self, B, T_q, T_k, device):
        mask = torch.tril(torch.ones(T_q, T_k, device=device, dtype=torch.uint8))
        mask = mask.unsqueeze(0).expand(B, -1, -1)
        return mask

    def _key_padding_mask(self, lengths: torch.Tensor, T: int):
        idxs = torch.arange(T, device=lengths.device).unsqueeze(0)
        return (idxs < lengths.unsqueeze(1))  # bool mask

    # ------------- static encoding with VSN ------------- #

    def _encode_static(
        self,
        static_cont: Optional[torch.Tensor],
        static_cat: Optional[torch.Tensor],
    ):
        """
        static_cont: (B, n_cont)
        static_cat:  (B, n_cat)
        Returns:
          static_ctx:   (B,d_model)   - aggregated static representation
          c_sel:        (B,d_model)
          c_enr:        (B,d_model)
          c_ctx:        (B,d_model)
          static_alpha: (B, V_static) - per-static-variable importance
        """
        B = static_cont.size(0) if static_cont is not None else static_cat.size(0)

        xs: List[torch.Tensor] = []

        # continuous statics -> each as (B,1,1)
        if static_cont is not None and self.static_cont_dim > 0:
            for i in range(self.static_cont_dim):
                xi = static_cont[:, i:i+1]           # (B,1)
                xi = xi.unsqueeze(1)                 # (B,1,1)
                xs.append(xi)

        # categorical statics -> embedding then (B,1,emb_dim)
        if self.static_cat_embs is not None and static_cat is not None:
            for j, emb in enumerate(self.static_cat_embs):
                ej = emb(static_cat[:, j])           # (B,emb_dim)
                ej = ej.unsqueeze(1)                 # (B,1,emb_dim)
                xs.append(ej)

        if not xs:
            # no static features at all -> dummy 0 vector
            dummy = torch.zeros(B, 1, 1, device=static_cont.device if static_cont is not None else static_cat.device)
            xs = [dummy]

        # static VSN over "time" dimension T=1
        s_seq, alpha, H = self.vsn_static(xs, ctx=None)  # s_seq: (B,1,d_model), alpha: (B,1,V)
        s = s_seq[:, 0, :]                               # (B,d_model)
        static_alpha = alpha[:, 0, :]                    # (B,V_static)

        # derive TFT's three static context vectors
        c_sel = self.c_selection(s)   # context for encoder/decoder VSNs
        c_enr = self.c_enrich(s)      # context for static enrichment
        c_ctx = self.c_context(s)     # context for post-LSTM gating

        return s, c_sel, c_enr, c_ctx, static_alpha

    # ------------- forward ------------- #

    def forward(
        self,
        enc_vars: Dict[str, torch.Tensor],   # each (B,T_enc,dim_i)
        dec_vars: Dict[str, torch.Tensor],   # each (B,T_dec,dim_j)
        enc_lengths: torch.Tensor,           # (B,)
        static_cont: Optional[torch.Tensor] = None,  # (B,n_cont)
        static_cat: Optional[torch.Tensor] = None,   # (B,n_cat)
    ):
        """
        Returns:
          (logits, eta_q), extras
          logits: (B,T_dec,num_classes) or (B,num_classes) if pred_len=1 squashed later
          eta_q:  (B,T_dec,Q) or (B,Q) if pred_len=1, or None
          extras: dict with
            - "enc_alpha":   (B,T_enc,V_enc)
            - "dec_alpha":   (B,T_dec,V_dec)
            - "static_alpha":(B,V_static)
            - "attn_weights":(B,T_dec,num_heads,T_enc)
        """
        device = enc_lengths.device
        B = enc_lengths.size(0)
        T_enc = next(iter(enc_vars.values())).size(1)
        T_dec = next(iter(dec_vars.values())).size(1)

        # ----- static encoding (with VSN) -----
        static_ctx, c_sel, c_enr, c_ctx, static_alpha = self._encode_static(static_cont, static_cat)

        # ----- encoder/decoder variable selection -----
        enc_list = [enc_vars[name] for name in self.enc_names]
        dec_list = [dec_vars[name] for name in self.dec_names]

        enc_sel, enc_alpha, _ = self.vsn_enc(enc_list, c_sel)  # (B,T_enc,d), (B,T_enc,V_enc)
        dec_sel, dec_alpha, _ = self.vsn_dec(dec_list, c_sel)  # (B,T_dec,d), (B,T_dec,V_dec)

        # ----- LSTMs -----
        enc_pack = nn.utils.rnn.pack_padded_sequence(
            enc_sel,
            enc_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        enc_out_pack, (h, c) = self.enc_lstm(enc_pack)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(
            enc_out_pack,
            batch_first=True,
            total_length=T_enc,
        )

        dec_out, _ = self.dec_lstm(dec_sel, (h, c))  # (B,T_dec,d_model)

        # ----- post-LSTM gating & static enrichment -----
        dec_fused = self.post_lstm_gating(dec_out, c_ctx)         # (B,T_dec,d)
        dec_enriched = self.static_enrichment(dec_fused, c_enr)   # (B,T_dec,d)

        # ----- attention over encoder -----
        enc_keep = self._key_padding_mask(enc_lengths, T_enc).unsqueeze(1)  # (B,1,T_enc)
        attn_out, attn_w = self.attn(
            q=dec_enriched,
            k=enc_out,
            v=enc_out,
            attn_mask=enc_keep,
        )  # attn_out: (B,T_dec,d)

        dec_after_attn = self.attn_gate_norm(attn_out, dec_enriched)  # (B,T_dec,d)
        dec_final = self.pos_wise_ffn(dec_after_attn)                 # (B,T_dec,d)

        # ----- heads -----
        if self.pred_len == 1:
            # use last decoder step only
            last = dec_final[:, -1, :]               # (B,d)
            logits = self.classifier(last)           # (B,num_classes)
            eta_q = self.eta_head(last) if self.eta_head is not None else None
        else:
            logits = self.classifier(dec_final)      # (B,T_dec,num_classes)
            eta_q = self.eta_head(dec_final) if self.eta_head is not None else None

        extras = {
            "enc_alpha": enc_alpha,        # (B,T_enc,V_enc)
            "dec_alpha": dec_alpha,        # (B,T_dec,V_dec)
            "static_alpha": static_alpha,  # (B,V_static)
            "attn_weights": attn_w,        # (B,T_dec,num_heads,T_enc)
        }

        return (logits, eta_q), extras
