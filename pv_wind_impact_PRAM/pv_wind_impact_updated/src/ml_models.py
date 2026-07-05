"""
src/ml_models.py
================
The machine-learning layer: a catalogue of regression algorithms, a single
training routine, evaluation metrics, and a multi-model benchmark.

Algorithms (all driven by the 100-parameter feature table):

    Tree ensembles      Random Forest, Extra Trees, Gradient Boosting,
                        Hist Gradient Boosting, Decision Tree, XGBoost
    Linear family       Linear, Ridge, Lasso, Elastic Net
    Distance / kernel   K-Nearest Neighbors, Support Vector Regression
    Sklearn neural     Neural Net (MLP), Deep NN, Wide NN
    PyTorch deep        1D CNN, Simple RNN, LSTM, GRU, BiLSTM,
                        Transformer, Autoencoder, RBFN, GAN, DBN,
                        Capsule Network, SOM

Models that are scale-sensitive are wrapped in a StandardScaler pipeline.
LightGBM and PyTorch models are added automatically when their packages
are installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

import config

# XGBoost is optional: if it (or its native lib) fails to import, the whole
# ml_models module must NOT crash — that would take down the entire app. We
# detect it here and register the model conditionally below (same pattern as
# LightGBM / the optional PyTorch models).
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None
    HAS_XGBOOST = False

RS = config.RANDOM_STATE


def _scaled(estimator) -> Pipeline:
    """Wrap a scale-sensitive estimator in a StandardScaler pipeline."""
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


# ===================================================================
# PyTorch Deep-Learning Models  (optional — requires torch)
# ===================================================================
class _PyTorchRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-compatible PyTorch regression model for tabular data.

    Parameters
    ----------
    architecture : str
        One of ``'cnn'``, ``'rnn'``, ``'lstm'``, ``'gru'``, ``'bilstm'``,
        ``'transformer'``, ``'autoencoder'``, ``'rbfn'``, ``'gan'``,
        ``'dbn'``, ``'capsule'``.
    input_dim : int
    """

    def __init__(self, architecture: str = "cnn", input_dim: int = 100):
        self.architecture = architecture
        self.input_dim = input_dim
        self.model_ = None
        self.device_ = None

    def get_params(self, deep=True):
        return {"architecture": self.architecture, "input_dim": self.input_dim}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ---- build --------------------------------------------------------------
    def _build_net(self):
        import torch.nn as nn
        d = self.input_dim
        arch = self.architecture.lower()

        if arch == "cnn":
            return _CNNNet(d)
        if arch == "rnn":
            return _RNNNet(d)
        if arch == "lstm":
            return _LSTMNet(d)
        if arch == "gru":
            return _GRUNet(d)
        if arch == "bilstm":
            return _BiLSTMNet(d)
        if arch == "transformer":
            return _TransformerNet(d)
        if arch == "rbfn":
            return _RBFNet(d)
        if arch == "capsule":
            return _CapsuleNet(d)
        if arch == "autoencoder":
            return _AutoencoderReg(d)
        if arch == "gan":
            return _GANReg(d)
        if arch == "dbn":
            return _DBNet(d)
        raise ValueError(f"Unknown architecture '{arch}'")

    # ---- fit ----------------------------------------------------------------
    def fit(self, X, y, **kwargs):
        import torch
        import torch.nn as nn

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
        y_t = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1),
                           device=self.device_)
        arch = self.architecture.lower()

        if arch == "gan":
            self._fit_gan(X_t, y_t, **kwargs)
        elif arch == "autoencoder":
            self._fit_autoencoder(X_t, y_t, **kwargs)
        else:
            net = self._build_net().to(self.device_)
            epochs = kwargs.get("epochs", 80)
            lr = kwargs.get("lr", 0.001)
            batch_size = kwargs.get("batch_size", 64)
            optim = torch.optim.Adam(net.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            n = len(X_t)
            for _ in range(epochs):
                idx = torch.randperm(n)
                for start in range(0, n, batch_size):
                    bx = X_t[idx[start:start + batch_size]]
                    by = y_t[idx[start:start + batch_size]]
                    optim.zero_grad()
                    loss = loss_fn(net(bx), by)
                    loss.backward()
                    optim.step()
            self.model_ = net
        return self

    def _fit_autoencoder(self, X_t, y_t, **kwargs):
        import torch
        import torch.nn as nn
        d = self.input_dim
        epochs_ae = kwargs.get("ae_epochs", 40)
        epochs_reg = kwargs.get("epochs", 60)
        batch_size = kwargs.get("batch_size", 64)
        lr = kwargs.get("lr", 0.001)

        # Encoder.
        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d, 64), nn.ReLU(), nn.BatchNorm1d(64),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 16), nn.ReLU(),
                )
            def forward(self, x): return self.net(x)

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(16, 32), nn.ReLU(),
                    nn.Linear(32, 64), nn.ReLU(),
                    nn.Linear(64, d),
                )
            def forward(self, x): return self.net(x)

        enc = _Encoder().to(self.device_)
        dec = _Decoder().to(self.device_)
        optim_ae = torch.optim.Adam(
            list(enc.parameters()) + list(dec.parameters()), lr=lr)
        loss_fn = nn.MSELoss()
        n = len(X_t)

        for _ in range(epochs_ae):
            idx = torch.randperm(n)
            for start in range(0, n, batch_size):
                bx = X_t[idx[start:start + batch_size]]
                optim_ae.zero_grad()
                recon = dec(enc(bx))
                loss = loss_fn(recon, bx)
                loss.backward()
                optim_ae.step()

        # Regression head on encoder.
        class _RegModel(nn.Module):
            def __init__(self, enc):
                super().__init__()
                self.enc = enc
                self.head = nn.Sequential(
                    nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
            def forward(self, x): return self.head(self.enc(x))

        net = _RegModel(enc).to(self.device_)
        optim_r = torch.optim.Adam(net.parameters(), lr=lr * 0.5)
        for _ in range(epochs_reg):
            idx = torch.randperm(n)
            for start in range(0, n, batch_size):
                bx = X_t[idx[start:start + batch_size]]
                by = y_t[idx[start:start + batch_size]]
                optim_r.zero_grad()
                loss = loss_fn(net(bx), by)
                loss.backward()
                optim_r.step()
        self.model_ = net

    def _fit_gan(self, X_t, y_t, **kwargs):
        import torch
        import torch.nn as nn
        d = self.input_dim
        noise_dim = 16
        batch_size = kwargs.get("batch_size", 64)
        epochs = kwargs.get("epochs", 60)
        n = len(X_t)

        class _Gen(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d + noise_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
                    nn.Linear(128, 64), nn.ReLU(),
                    nn.Linear(64, 1),
                )
            def forward(self, noise, feat):
                return self.net(torch.cat([noise, feat], dim=1))

        class _Disc(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d + 1, 128), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Dropout(0.3),
                    nn.Linear(64, 1), nn.Sigmoid(),
                )
            def forward(self, feat, target):
                return self.net(torch.cat([feat, target], dim=1))

        gen = _Gen().to(self.device_)
        disc = _Disc().to(self.device_)
        opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002)
        opt_d = torch.optim.Adam(disc.parameters(), lr=0.0002)
        bce = nn.BCELoss()

        for _ in range(epochs):
            idx = torch.randperm(n)
            for start in range(0, n, batch_size):
                bx = X_t[idx[start:start + batch_size]]
                by = y_t[idx[start:start + batch_size]]
                bs_actual = bx.size(0)
                half = max(1, bs_actual // 2)
                # Train discriminator.
                noise = torch.randn(half, noise_dim, device=self.device_)
                fake = gen(noise, bx[:half]).detach()
                real_pred = disc(bx[:half], by[:half])
                fake_pred = disc(bx[:half], fake)
                loss_d = bce(real_pred, torch.ones_like(real_pred)) + \
                         bce(fake_pred, torch.zeros_like(fake_pred))
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
                # Train generator.
                noise2 = torch.randn(bs_actual, noise_dim, device=self.device_)
                fake2 = gen(noise2, bx)
                validity = disc(bx, fake2)
                loss_g = bce(validity, torch.ones_like(validity))
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
        self.model_ = gen
        self._noise_dim = noise_dim

    # ---- predict ------------------------------------------------------------
    def predict(self, X):
        import torch
        if self.model_ is None:
            return np.zeros(X.shape[0])
        self.model_.eval()
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device_)
        with torch.no_grad():
            arch = self.architecture.lower()
            if arch == "gan":
                noise = torch.randn(X_t.size(0), 16, device=self.device_)
                pred = self.model_(noise, X_t)
            else:
                pred = self.model_(X_t)
        return pred.cpu().numpy().flatten()


# ---- PyTorch network definitions -------------------------------------------
# torch is OPTIONAL. If it isn't installed the module must still import cleanly
# (the app depends on it); the deep-learning models simply aren't registered in
# build_model_catalogue(). We provide a tiny shim so the nn.Module subclasses
# below still *define* without torch — their bodies only touch nn.Conv1d/etc.
# at instantiation time, which never happens when the models aren't registered.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    torch = None
    F = None
    _HAS_TORCH = False

    class _NNShim:
        Module = object

        def __getattr__(self, name):
            raise ImportError(
                "PyTorch is required for this model but is not installed")

    nn = _NNShim()


class _CNNNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = self.drop(x)
        return self.fc(x)


class _RNNNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.rnn = nn.RNN(1, 64, batch_first=True, nonlinearity="tanh")
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.rnn(x)
        x = self.drop(x[:, -1, :])
        return self.fc(x)


class _LSTMNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 64, batch_first=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x[:, :, :])
        x = self.drop(x[:, -1, :])
        return self.fc(x)


class _GRUNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gru1 = nn.GRU(1, 64, batch_first=True)
        self.gru2 = nn.GRU(64, 32, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.gru1(x)
        x, _ = self.gru2(x[:, :, :])
        x = self.drop(x[:, -1, :])
        return self.fc(x)


class _BiLSTMNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.drop(x[:, -1, :])
        return self.fc(x)


class _TransformerNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 64)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = self.proj(x).unsqueeze(1)
        x = self.encoder(x)
        x = x.squeeze(1)
        x = self.drop(x)
        return self.fc(x)


class _RBFNet(nn.Module):
    def __init__(self, input_dim, n_centers=64):
        super().__init__()
        self.n_centers = n_centers
        self.centers = nn.Parameter(torch.randn(n_centers, input_dim) * 0.1)
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(n_centers, 1)

    def forward(self, x):
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist = (diff ** 2).sum(-1)
        rbf = torch.exp(-self.gamma * dist)
        rbf = self.drop(rbf)
        return self.fc(rbf)


class _CapsuleNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.primary = nn.Linear(128, 16 * 8)
        self.flatten = nn.Flatten()
        self.fc2 = nn.Linear(16 * 8, 64)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    @staticmethod
    def squash(x, dim=-1):
        s = (x ** 2).sum(dim=dim, keepdim=True)
        return (s / (1 + s)) * (x / (s.sqrt() + 1e-9))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.primary(x)
        x = x.view(x.size(0), 16, 8)
        x = self.squash(x)
        x = self.flatten(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.out(x)


class _AutoencoderReg(nn.Module):
    """Placeholder — actual training is handled in _PyTorchRegressor._fit_autoencoder."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class _GANReg(nn.Module):
    """Placeholder — actual training is handled in _PyTorchRegressor._fit_gan."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


class _DBNet(nn.Module):
    """Deep Belief Network via stacked encoder layers + fine-tuning head."""
    def __init__(self, input_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Sigmoid(),
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 32), nn.Sigmoid(),
        )
        self.reg = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.enc(x)
        return self.reg(x)


# ===================================================================
# SOM Regressor (Self-Organizing Map, numpy-only)
# ===================================================================
class _SOMRegressor(BaseEstimator, RegressorMixin):
    """Self-Organizing Map adapted for regression."""

    def __init__(self, map_size: int = 10, sigma: float = 1.5,
                 learning_rate: float = 0.5, n_iter: int = 5000):
        self.map_size = map_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights_ = None
        self.target_map_ = None
        self._fitted = False

    def get_params(self, deep=True):
        return {"map_size": self.map_size, "sigma": self.sigma,
                "learning_rate": self.learning_rate, "n_iter": self.n_iter}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def _bmu(self, x):
        diffs = self.weights_ - x.reshape(1, 1, -1)
        dists = np.sum(diffs ** 2, axis=2)
        return np.unravel_index(np.argmin(dists), dists.shape)

    def fit(self, X, y):
        rng = np.random.RandomState(RS)
        n_features = X.shape[1]
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).flatten()
        self.weights_ = rng.randn(self.map_size, self.map_size, n_features).astype(np.float32)
        std = X_arr.std(axis=0).clip(1e-9)
        X_norm = (X_arr - X_arr.mean(axis=0)) / std
        self._mean = X_arr.mean(axis=0).astype(np.float32)
        self._std = std.astype(np.float32)
        t_max = float(self.n_iter)
        for t in range(self.n_iter):
            lr = self.learning_rate * (1 - t / t_max)
            sig = self.sigma * (1 - t / t_max)
            idx = rng.randint(len(X_norm))
            x_sample = X_norm[idx]
            bmu = self._bmu(x_sample)
            for i in range(self.map_size):
                for j in range(self.map_size):
                    dist = (i - bmu[0]) ** 2 + (j - bmu[1]) ** 2
                    neigh = np.exp(-dist / (2 * sig ** 2))
                    self.weights_[i, j] += lr * neigh * (x_sample - self.weights_[i, j])
        X_norm_t = (X_arr - self._mean) / self._std
        self.target_map_ = np.zeros((self.map_size, self.map_size), dtype=np.float32)
        counts = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        for i in range(len(X_norm_t)):
            bmu = self._bmu(X_norm_t[i])
            self.target_map_[bmu] += y_arr[i]
            counts[bmu] += 1
        counts = np.maximum(counts, 1)
        self.target_map_ /= counts
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            return np.zeros(X.shape[0])
        X_arr = np.asarray(X, dtype=np.float32)
        X_norm = (X_arr - self._mean) / self._std
        preds = np.zeros(len(X_norm))
        for i in range(len(X_norm)):
            bmu = self._bmu(X_norm[i])
            preds[i] = self.target_map_[bmu]
        return preds


# ===================================================================
# Model catalogue
# ===================================================================
def build_model_catalogue() -> dict:
    """Return {display_name: factory()} for every available algorithm."""
    catalogue = {
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=None, random_state=RS, n_jobs=1),
        "Extra Trees": lambda: ExtraTreesRegressor(
            n_estimators=200, random_state=RS, n_jobs=1),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RS),
        "Hist Gradient Boosting": lambda: HistGradientBoostingRegressor(
            max_iter=300, random_state=RS),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=RS),
        "Linear Regression": lambda: _scaled(LinearRegression()),
        "Ridge": lambda: _scaled(Ridge(alpha=1.0, random_state=RS)),
        "Lasso": lambda: _scaled(Lasso(alpha=0.01, max_iter=5000, random_state=RS)),
        "Elastic Net": lambda: _scaled(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RS)),
        "K-Nearest Neighbors": lambda: _scaled(KNeighborsRegressor(n_neighbors=12)),
        "Support Vector Regression": lambda: _scaled(
            SVR(kernel="rbf", C=20.0, epsilon=0.5)),
        "Neural Net (MLP)": lambda: _scaled(MLPRegressor(
            hidden_layer_sizes=(128, 64), activation="relu", max_iter=400,
            early_stopping=True, random_state=RS)),
        "Deep Neural Net": lambda: _scaled(MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32), activation="relu",
            max_iter=500, alpha=0.001, early_stopping=True, random_state=RS)),
        "Wide Neural Net": lambda: _scaled(MLPRegressor(
            hidden_layer_sizes=(512, 256), activation="relu",
            max_iter=500, alpha=0.001, early_stopping=True, random_state=RS)),
    }

    # Optional XGBoost (registered only if it imported cleanly above).
    if HAS_XGBOOST:
        catalogue["XGBoost"] = lambda: XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=7,
            subsample=0.9, colsample_bytree=0.9, random_state=RS, n_jobs=1)

    # Optional LightGBM.
    try:
        from lightgbm import LGBMRegressor
        catalogue["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=48,
            subsample=0.9, random_state=RS, n_jobs=1, verbose=-1)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # PyTorch deep-learning models (optional)
    # ------------------------------------------------------------------
    try:
        import torch  # noqa: F401

        DL_ARCHITECTURES = [
            ("cnn",         "1D CNN"),
            ("rnn",         "Simple RNN"),
            ("lstm",        "LSTM"),
            ("gru",         "GRU"),
            ("bilstm",      "BiLSTM"),
            ("transformer", "Transformer"),
            ("rbfn",        "RBF Network"),
            ("capsule",     "Capsule Network"),
            ("autoencoder", "Autoencoder"),
            ("gan",         "GAN"),
            ("dbn",         "Deep Belief Network"),
        ]
        for arch_id, arch_label in DL_ARCHITECTURES:
            catalogue[arch_label] = lambda a=arch_id: _scaled(
                _PyTorchRegressor(architecture=a))
    except Exception:
        pass

    # SOM (always available).
    catalogue["Self Organizing Map"] = lambda: _scaled(_SOMRegressor())

    return catalogue


MODEL_CATALOGUE = build_model_catalogue()
MODEL_NAMES = list(MODEL_CATALOGUE.keys())


@dataclass
class TrainResult:
    model: object
    model_name: str
    target: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    metrics: dict


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mean_actual": float(np.mean(y_true)),
    }


def train_model(df: pd.DataFrame, target: str, model_name: str,
                test_size: float = 0.2, chronological: bool = True) -> TrainResult:
    """Train one algorithm on the 100 features to predict one target.

    The feature table is an hourly time series with lag / rolling-window and
    cyclical-time features, so a *shuffled* split would scatter temporally
    adjacent, near-duplicate rows across train and test and leak information —
    inflating R^2 versus any real forecasting use. We therefore default to a
    ``chronological`` split (train on the earlier rows, test on the most recent
    ``test_size`` fraction). Pass ``chronological=False`` for the old shuffled
    behaviour.
    """
    X = df[config.FEATURES]
    y = df[target]
    if chronological:
        n_test = max(1, int(round(len(X) * test_size)))
        X_train, X_test = X.iloc[:-n_test], X.iloc[-n_test:]
        y_train, y_test = y.iloc[:-n_test], y.iloc[-n_test:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RS)

    model = MODEL_CATALOGUE[model_name]()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return TrainResult(
        model=model, model_name=model_name, target=target,
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        y_pred=y_pred, metrics=_metrics(y_test, y_pred),
    )


def benchmark_models(df: pd.DataFrame, target: str,
                     model_names: list[str] | None = None,
                     test_size: float = 0.2) -> pd.DataFrame:
    """
    Train every (selected) algorithm on a target and return a leaderboard
    sorted by R^2 descending.  Failures are reported as NaN rows rather than
    crashing the whole benchmark.
    """
    names = model_names or MODEL_NAMES
    rows = []
    for name in names:
        try:
            res = train_model(df, target, name, test_size=test_size)
            rows.append({
                "Model": name,
                "R2": res.metrics["r2"],
                "R2_pct": res.metrics["r2"] * 100.0,
                "MAE": res.metrics["mae"],
                "RMSE": res.metrics["rmse"],
            })
        except Exception as exc:
            rows.append({"Model": name, "R2": np.nan, "R2_pct": np.nan,
                         "MAE": np.nan, "RMSE": np.nan, "Error": str(exc)})
    out = pd.DataFrame(rows).sort_values("R2", ascending=False, na_position="last")
    return out.reset_index(drop=True)
