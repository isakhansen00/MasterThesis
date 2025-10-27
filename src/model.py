import math
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random


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
    def __init__(self, d_in, d_hidden, d_out=None, ctx_dim: Optional[int]=None, dropout=0.1):
        super().__init__()
        d_out = d_out or d_in
        self.w2 = nn.Linear(d_in, d_hidden)
        self.w3 = nn.Linear(ctx_dim, d_hidden, bias=False) if ctx_dim is not None else None
        self.elu = nn.ELU()
        self.w1 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.gate_add_norm = GateAddNorm(d_out)

    def forward(self, x, ctx: Optional[torch.Tensor]=None):
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
    
    def forward(self, query, key, value, mask: Optional[torch.Tensor]=None):
        dk = query.size(-1)
        attn = torch.bmm(query, key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        w = F.softmax(attn, dim=-1)
        w = self.dropout(w)
        out = torch.bmm(w, value)
        return out, w


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads=4, dropout=0.1):
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
    
    def forward(self, q, k, v, attn_mask: Optional[torch.Tensor]=None):
        V = self.v_proj(v)
        heads, weights = [], []
        for i in range(self.n):
            Qi = self.q_proj[i](q)
            Ki = self.k_proj[i](k)
            out, w = self.attn(Qi, Ki, V, attn_mask)
            heads.append(out)
            weights.append(w)
        H = torch.stack(heads, dim=2)
        W = torch.stack(weights, dim=2)
        H = H.mean(dim=2)
        out = self.out(H)
        out = self.drop(out)
        return out, W
    

class VariableSelectionNetwork(nn.Module):
    def __init__(self, var_dims, d_model, ctx_dim: Optional[int], dropout=0.1):
        super().__init__()
        self.V = len(var_dims)
        self.proj = nn.ModuleList([nn.Linear(dv, d_model) for dv in var_dims])
        self.var_grn = nn.ModuleList([GRN(d_model, d_model, d_model, ctx_dim, dropout) for _ in var_dims])
        self.flatten_grn = GRN(self.V * d_model, d_model, self.V, ctx_dim, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, xs: torch.Tensor, ctx: Optional[torch.Tensor]):
        Hs = []
        for i, x in enumerate(xs):
            h = self.proj[i](x)
            h = self.var_grn[i](h, ctx)
            Hs.append(h)
        H = torch.stack(Hs, dim=2)
        flat = H.flatten(start_dim=2)
        scores = self.flatten_grn(flat, ctx)
        alpha = self.softmax(scores)
        y = torch.sum(alpha.unsqueeze(-1) * H, dim=2)
        return y, alpha, H



class TFTClassifier(nn.Module):
    def __init__(
            
            self,
            enc_var_dims: Dict[str, int],
            dec_var_dims: Dict[str, int],
            static_cont_dim: int = 0,
            static_cat_cardinalities: Optional[list] = None,
            static_cat_emb_dim: int = 16,
            
            d_model: int = 128,
            lstm_layers: int = 1,
            dropout: float = 0.1,
            num_heads: int = 4,
            
            num_classes: int = 20,
            pred_len: int = 5,
            eta_quantiles: Tuple[float,...] = (0.1, 0.5, 0.9),
            ):
        super().__init__()
        self.pred_len = pred_len
        self.d_model = d_model

        self.static_cat_embs = None
        static_dim = 0
        if static_cat_cardinalities:
            self.static_cat_embs = nn.ModuleList([nn.Embedding(c, static_cat_emb_dim) for c in static_cat_cardinalities])
            static_dim += len(static_cat_cardinalities) * static_cat_emb_dim
        if static_cont_dim > 0:
            static_dim += static_cont_dim
        
        self.static_proj = nn.Linear(max(1, static_dim), d_model)

        self.c_selection = GRN(d_model, d_model, d_model, dropout=dropout)
        self.c_enrich = GRN(d_model, d_model, d_model, dropout=dropout)
        self.c_context = GRN(d_model, d_model, d_model, dropout=dropout)


        self.enc_names = list(enc_var_dims.keys())
        self.dec_names = list(dec_var_dims.keys())
        self.vsn_enc = VariableSelectionNetwork([enc_var_dims[k] for k in self.enc_names], d_model, ctx_dim=d_model, dropout=dropout)
        self.vsn_dec = VariableSelectionNetwork([dec_var_dims[k] for k in self.dec_names], d_model, ctx_dim=d_model, dropout=dropout)
        self.vsn_static = VariableSelectionNetwork([d_model], d_model, ctx_dim=d_model, dropout=dropout)

        self.enc_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.dec_lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)

        self.post_lstm_gating = GRN(d_model, d_model, d_model, ctx_dim=d_model, dropout=dropout)
        self.static_enrichment = GRN(d_model, d_model, d_model, ctx_dim=d_model, dropout=dropout)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.attn_gate_norm = GateAddNorm(d_model)
        self.pos_wise_ffn = GRN(d_model, d_model, d_model, dropout=dropout)



        self.classifier = nn.Linear(d_model, num_classes)


        self.eta_quantiles = eta_quantiles
        self.eta_head = nn.Linear(d_model, len(eta_quantiles)) if len(eta_quantiles) > 0 else None

    
    def _make_causal_mask(self, B, T_q, T_k, device):
        mask = torch.tril(torch.ones(T_q, T_k, device=device, dtype=torch.uint8)).unsqueeze(0).expand(B, -1, -1)
        return mask

    def _key_padding_mask(self, lengths: torch.Tensor, T: int):
        idxs = torch.arange(T, device=lengths.device).unsqueeze(0)
        return idxs < lengths.unsqueeze(1)

    def _encode_static(self, static_cont: Optional[torch.Tensor], static_cat: Optional[torch.Tensor]):
        pieces = []
        if self.static_cat_embs is not None and static_cat is not None:
            embs = [emb(static_cat[:, i]) for i, emb in enumerate(self.static_cat_embs)]
            pieces.append(torch.cat(embs, dim=-1))
        if static_cont is not None and static_cont.size(-1) > 0:
            pieces.append(static_cont)
        if len(pieces) == 0:
            B = static_cat.size(0) if static_cat is not None else static_cont.size(0)
            x = torch.zeros(B, 1, device=static_cat.device if static_cat is not None else static_cont.device)
        else:
            x = torch.cat(pieces, dim=-1)
        s = self.static_proj(x)  # (B,d)
        c_sel = self.c_selection(s)  # (B,d)
        c_enr = self.c_enrich(s)    # (B,d)
        c_ctx = self.c_context(s)   # (B,d)
        return c_sel, c_enr, c_ctx

    def forward(
            self, 
            enc_vars: Dict[str, torch.Tensor],
            dec_vars: Dict[str, torch.Tensor],
            enc_lengths: torch.Tensor,
            static_cont: Optional[torch.Tensor]=None,
            static_cat: Optional[torch.Tensor]=None,
            ):
        B = next(iter(enc_vars.values())).size(0)
        T_enc = next(iter(enc_vars.values())).size(1)
        T_dec = next(iter(dec_vars.values())).size(1)
        device = enc_lengths.device

        c_sel, c_enr, c_ctx = self._encode_static(static_cont, static_cat)

        enc_list = [enc_vars[name] for name in self.enc_names]
        dec_list = [dec_vars[name] for name in self.dec_names]

        enc_sel, enc_alpha = self.vsn_enc(enc_list, c_sel)
        dec_sel, dec_alpha = self.vsn_dec(dec_list, c_sel)

        enc_pack = nn.utils.rnn.pack_padded_sequence(enc_sel, enc_lengths.cpu(), batch_first=True, enforce_sorted=False)
        enc_out_pack, (h,c) = self.enc_lstm(enc_pack)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out_pack, batch_first=True, total_length=T_enc)

        dec_out, _ = self.dec_lstm(dec_sel, (h,c))

        dec_fused = self.post_lstm_gating(dec_out, c_ctx)

        dec_enriched = self.static_enrichment(dec_fused, c_enr)

        enc_keep = self._key_padding_mask(enc_lengths, T_enc).unsqueeze(1) 
        attn_out, attn_w = self.attn(q=dec_enriched, k=enc_out, v=enc_out, attn_mask=enc_keep)

        dec_after_attn = self.attn_gate_norm(attn_out, dec_enriched)

        dec_final = self.pos_wise_ffn(dec_after_attn)

        if self.pred_len == 1:
            logits = self.classifier(dec_final[:, -1, :]) 
            eta_q = self.eta_head(dec_final[:, -1, :]) if self.eta_head is not None else None
        else:
            logits = self.classifier(dec_final)  
            eta_q = self.eta_head(dec_final) if self.eta_head is not None else None

        return (logits, eta_q), {
            "enc_alpha": enc_alpha,      
            "dec_alpha": dec_alpha,       
            "attn_weights": attn_w,       
        }    



NUM_PORTS = 5


def pinball_loss(pred_q: torch.Tensor, y: torch.Tensor, quantiles: Tuple[float,...]):
    """
    pred_q: [B, T, Q] or [B, Q]
    y:      [B, T]     or [B]
    """
    if pred_q.dim() == 2:  # [B,Q] -> [B,1,Q], [B] -> [B,1]
        pred_q = pred_q.unsqueeze(1)
        y = y.unsqueeze(1)

    y = y.unsqueeze(-1).expand_as(pred_q)   # [B,T,Q]
    e = y - pred_q
    losses = []
    for qi, q in enumerate(quantiles):
        losses.append(torch.max((q-1)*e[..., qi], q*e[..., qi]).mean())
    return sum(losses) / len(losses)


def one_hot(idx: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(idx, num_classes=num_classes).float()

class FakeAISDataset(Dataset):
    """
    Generates synthetic AIS-like tracks with weak signal:
    - Lat/Lon random walks, heading (~cog) points toward one of NUM_PORTS.
    - Destination port chosen by latent quadrant/heading.
    - Future-known features: hour-of-day, day-of-week, wind forecast (random).
    """
    def __init__(self, n_samples=2000, t_enc=64, t_dec=3, device="cpu"):
        super().__init__()
        self.n = n_samples
        self.T_enc = t_enc
        self.T_dec = t_dec
        self.device = device

        # define 5 "port" centroids
        self.port_xy = torch.tensor([
            [ 10.0,  60.0],  # Port A
            [ 10.1,  60.2],  # Port B
            [ 10.2,  60.4],  # Port C
            [ 10.3,  60.1],  # Port D
            [ 10.4,  60.3],  # Port E
        ], dtype=torch.float)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        # static features
        vessel_type = torch.randint(0, 6, (1,))     # 6 types
        loa = torch.rand(1) * 100 + 10              # 10..110
        gt  = torch.rand(1) * 3000 + 100            # 100..3100

        # pick a true destination port
        dest = torch.randint(0, NUM_PORTS, (1,)).item()
        dest_xy = self.port_xy[dest]

        # starting point near ports
        lat0 = 59.8 + torch.rand(1) * 0.6
        lon0 = 9.8  + torch.rand(1) * 0.6

        # encode window: random walk moving roughly toward destination
        lat = [lat0]
        lon = [lon0]
        sog = []
        cog = []

        for t in range(self.T_enc):
            vec = (dest_xy - torch.tensor([lon[-1], lat[-1]]))
            dx, dy = vec[0].item(), vec[1].item()
            # heading (cog) ~ angle toward dest + noise
            heading = math.atan2(dy, dx) + (random.random() - 0.5) * 0.2
            speed = 8 + random.random() * 4  # 8..12 kn
            # move a tiny step
            lon_new = lon[-1] + math.cos(heading) * 0.01 + (random.random()-0.5)*0.005
            lat_new = lat[-1] + math.sin(heading) * 0.01 + (random.random()-0.5)*0.005
            lon.append(torch.tensor(lon_new))
            lat.append(torch.tensor(lat_new))
            sog.append(torch.tensor([speed], dtype=torch.float))
            cog.append(torch.tensor([heading], dtype=torch.float))

        # trim first element to keep lengths aligned
        lon = torch.stack(lon[1:], dim=0).unsqueeze(-1)  # (T_enc,1)
        lat = torch.stack(lat[1:], dim=0).unsqueeze(-1)  # (T_enc,1)
        sog = torch.stack(sog, dim=0)                    # (T_enc,1)
        cog = torch.stack(cog, dim=0)                    # (T_enc,1)


        start_hour = torch.randint(0, 24, (1,)).item()
        hours = (torch.arange(self.T_dec) + start_hour) % 24
        dows  = torch.randint(0, 7, (self.T_dec,))
        hour_oh = one_hot(hours, 24)                     
        dow_oh  = one_hot(dows, 7)                       
        wind_fc = torch.randn(self.T_dec, 1) * 0.5 + 5.0 


        y = torch.full((self.T_dec,), dest, dtype=torch.long)

        DT_MIN = 5  

        pos_lon, pos_lat = lon[-1].item(), lat[-1].item()
        steps_to_arrival = 0
        while True:
            vec = (dest_xy - torch.tensor([pos_lon, pos_lat]))
            dx, dy = vec[0].item(), vec[1].item()
            if (dx*dx + dy*dy) ** 0.5 < 0.02: 
                break
            heading = math.atan2(dy, dx)
            speed = 9.0 
            pos_lon += math.cos(heading) * 0.01
            pos_lat += math.sin(heading) * 0.01
            steps_to_arrival += 1

        tta_minutes = []
        for k in range(self.T_dec):
            rem = max(steps_to_arrival - k, 0) * DT_MIN
            tta_minutes.append(rem)
        eta = torch.tensor(tta_minutes, dtype=torch.float)  # (T_dec,)

        sample = {
            "enc_vars": {
                "lat": lat,         # (T_enc,1)
                "lon": lon,         # (T_enc,1)
                "sog": sog,         # (T_enc,1)
                "cog": cog,         # (T_enc,1)
            },
            "dec_vars": {
                "hour": hour_oh,    # (T_dec,24)
                "dow": dow_oh,      # (T_dec,7)
                "wind": wind_fc,    # (T_dec,1)
            },
            "enc_len": torch.tensor(self.T_enc, dtype=torch.long),
            "static_cat": vessel_type,               # (1,)
            "static_cont": torch.cat([loa, gt], -1), # (2,)
            "target": y,                             # (T_dec,)
            "eta_target": eta, 
        }
        return sample

def collate_batch(batch: List[Dict]):
    # All sequences are same length in this toy; just stack.
    B = len(batch)
    enc_vars = {}
    dec_vars = {}
    # gather keys from first element
    enc_keys = list(batch[0]["enc_vars"].keys())
    dec_keys = list(batch[0]["dec_vars"].keys())

    for k in enc_keys:
        enc_vars[k] = torch.stack([b["enc_vars"][k] for b in batch], dim=0)  # (B,T_enc,dim_i)
    for k in dec_keys:
        dec_vars[k] = torch.stack([b["dec_vars"][k] for b in batch], dim=0)  # (B,T_dec,dim_j)

    enc_lengths = torch.stack([b["enc_len"] for b in batch], dim=0)          # (B,)
    static_cat = torch.stack([b["static_cat"] for b in batch], dim=0)        # (B,1)
    static_cont = torch.stack([b["static_cont"] for b in batch], dim=0)      # (B,2)
    targets = torch.stack([b["target"] for b in batch], dim=0)               # (B,T_dec)
    eta_targets = torch.stack([b["eta_target"] for b in batch], dim=0)  # (B, T_dec)

    return enc_vars, dec_vars, enc_lengths, static_cont, static_cat, targets, eta_targets

# -------------------------
# (C) Wire it together and train
# -------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_ENC = 64
    T_DEC = 3
    BATCH = 64
    NUM_CLASSES = NUM_PORTS

    ds = FakeAISDataset(n_samples=2000, t_enc=T_ENC, t_dec=T_DEC, device=device)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate_batch)

    # define per-variable dims for enc/dec (must match dataset tensors)
    enc_var_dims = {"lat":1, "lon":1, "sog":1, "cog":1}
    dec_var_dims = {"hour":24, "dow":7, "wind":1}

    model = TFTClassifier(
        enc_var_dims=enc_var_dims,
        dec_var_dims=dec_var_dims,
        static_cont_dim=2,               # LOA, GT
        static_cat_cardinalities=[6],    # vessel_type has 6 categories
        static_cat_emb_dim=8,
        d_model=64,
        lstm_layers=1,
        dropout=0.1,
        num_heads=4,
        num_classes=NUM_CLASSES,
        pred_len=T_DEC,                  # seq-to-seq classification
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    for step, batch in enumerate(dl):
        enc_vars, dec_vars, enc_lengths, static_cont, static_cat, targets, eta_targets = batch
        # move to device
        enc_vars = {k: v.to(device) for k, v in enc_vars.items()}
        dec_vars = {k: v.to(device) for k, v in dec_vars.items()}
        enc_lengths = enc_lengths.to(device)
        static_cont = static_cont.to(device).float()
        static_cat = static_cat.to(device).long()
        targets = targets.to(device).long()  # (B,T_dec)

        # forward
        (logits, eta_q), extras = model(enc_vars, dec_vars, enc_lengths, static_cont, static_cat)
        # logits: (B,T_dec,NUM_CLASSES)

        # loss (seq-to-seq cross-entropy)
        ce = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES),   # (B*T_dec, C)
            targets.reshape(-1)                 # (B*T_dec,)
        )

        ql = pinball_loss(eta_q, eta_targets.to(device).float(), model.eta_quantiles)

        loss = 0.7 * ce + 0.3 * ql

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0:
            with torch.no_grad():
                pred = logits.argmax(-1)  # (B,T_dec)
                acc = (pred == targets).float().mean().item()
            print(f"step {step:04d} | loss {loss.item():.4f} | acc {acc:.3f}")

        if step == 100:  # tiny demo
            break

    # Example: inspect variable importances from last batch forward
    with torch.no_grad():
        (logits, eta_q), extras = model(enc_vars, dec_vars, enc_lengths, static_cont, static_cat)
        print("enc_alpha shape:", extras["enc_alpha"].shape)  # (B,T_enc,V_enc)
        print("dec_alpha shape:", extras["dec_alpha"].shape)  # (B,T_dec,V_dec)
        print("attn weights:", extras["attn_weights"].shape)  # (B,T_dec,T_enc,heads)

if __name__ == "__main__":
    main()