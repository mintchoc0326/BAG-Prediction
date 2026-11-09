"""GINE graph construction, independent early stopping and fixed-epoch refit."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPPER = np.triu_indices(84, 1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graphs(raw, values, threshold, transforms):
    """Topology uses per-scan raw magnitude; signed transformed edge attributes.

    No demographics in node/edge features. DWI topology requires positive raw
    edges. fMRI topology retains either sign. Fusion uses the union of modalities.
    Missing QC edges are excluded from topology and mapped to scaled zero.
    """
    output = []
    modalities = list(values)
    for row in range(len(next(iter(values.values())))):
        topology = np.zeros(3486, dtype=bool)
        matrices = []
        for m in modalities:
            magnitude = np.abs(raw[m][row])
            eligible = (magnitude > 0) & transforms[m].mask_
            if not eligible.any():
                continue
            cutoff = np.quantile(magnitude[eligible], 1 - threshold)
            topology |= eligible & (magnitude >= cutoff)
        if not topology.any():
            raise ValueError("Empty graph after topology selection")
        for m in modalities:
            matrix = np.zeros((84, 84), dtype=np.float32)
            matrix[UPPER] = values[m][row]
            matrix[(UPPER[1], UPPER[0])] = values[m][row]
            matrices.append(matrix)
        source, target = UPPER[0][topology], UPPER[1][topology]
        edge_index = np.vstack([np.r_[source, target], np.r_[target, source]])
        edge = np.column_stack(
            [
                np.r_[values[m][row, topology], values[m][row, topology]]
                for m in modalities
            ]
        )
        output.append(
            Data(
                x=torch.tensor(np.concatenate(matrices, axis=1)),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge, dtype=torch.float32),
            )
        )
    return output


class GINE(nn.Module):
    def __init__(self, node_dim, edge_dim, layers, width, residual=False, dropout=0.3):
        super().__init__()
        self.residual = residual
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(layers):
            size = node_dim if i == 0 else width
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(size, width), nn.ReLU(), nn.Linear(width, width)
                    ),
                    edge_dim=edge_dim,
                )
            )
            self.norms.append(nn.BatchNorm1d(width))
            self.skips.append(
                nn.Linear(size, width, bias=False) if residual else nn.Identity()
            )
        self.head = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, data, embedding=False):
        x = data.x
        for i, (conv, norm, skip) in enumerate(zip(self.convs, self.norms, self.skips)):
            out = norm(conv(x, data.edge_index, data.edge_attr))
            x = F.relu(out + skip(x) if self.residual else out)
            if i < len(self.convs) - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
        pooled = global_mean_pool(x, data.batch)
        return pooled if embedding else self.head(pooled).squeeze(-1)


def predict(model, items, mean, std, batch_size=32, embedding=False):
    model.eval()
    output = []
    with torch.no_grad():
        for batch in DataLoader(items, batch_size=batch_size, shuffle=False):
            output.append(
                model(batch.to(DEVICE), embedding=embedding).detach().cpu().numpy()
            )
    value = np.concatenate(output)
    # Age predictions must round-trip through CSV at full calibration precision.
    return value if embedding else value.astype(np.float64) * std + mean


def fit(train, age, config, settings, seed, validation=None, val_age=None, epochs=None):
    seed_all(seed)
    torch.set_num_threads(settings.get("threads", 4))
    mean = float(np.mean(age))
    std = max(float(np.std(age)), 1e-8)
    items = [g.clone() for g in train]
    for graph, target in zip(items, age):
        graph.y = torch.tensor([(target - mean) / std], dtype=torch.float32)
    model = GINE(
        items[0].x.shape[1],
        items[0].edge_attr.shape[1],
        config["layers"],
        config["width"],
        config.get("residual", False),
        settings.get("dropout", 0.3),
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.get("learning_rate", 5e-4),
        weight_decay=settings.get("weight_decay", 5e-4),
    )
    loader = DataLoader(items, batch_size=settings.get("batch_size", 32), shuffle=True)
    best = np.inf
    best_epoch = 0
    state = None
    history = []
    limit = int(epochs if epochs is not None else settings.get("max_epochs", 250))
    for epoch in range(1, limit + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            p = model(batch)
            loss = F.mse_loss(p, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch.num_graphs
            n += batch.num_graphs
        tr = predict(model, train, mean, std)
        row = dict(
            epoch=epoch,
            optimization_mse_z=total / n,
            train_mae=float(np.mean(abs(tr - age))),
            train_mse_z=float(np.mean(((tr - age) / std) ** 2)),
        )
        if validation is not None:
            va = predict(model, validation, mean, std)
            score = float(np.mean(abs(va - val_age)))
            row.update(
                validation_mae=score,
                validation_mse_z=float(np.mean(((va - val_age) / std) ** 2)),
            )
            if score < best:
                best = score
                best_epoch = epoch
                state = copy.deepcopy(model.state_dict())
            if not np.isfinite(score):
                raise ValueError("Nonfinite validation loss")
        history.append(row)
        if validation is not None and epoch - best_epoch >= settings.get(
            "patience", 25
        ):
            break
    if validation is not None:
        if state is None:
            raise ValueError("No finite model state")
        model.load_state_dict(state)
    else:
        best_epoch = limit
    return model, mean, std, best_epoch, history
