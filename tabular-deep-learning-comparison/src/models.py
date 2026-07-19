"""Model factories and lightweight wrappers for tabular prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    TABNET_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without tabnet.
    TabNetClassifier = None
    TabNetRegressor = None
    TABNET_AVAILABLE = False


class ClassificationModel(Protocol):
    """Common interface expected by the experiment runner."""

    model_name: str
    history: dict[str, Any]

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "ClassificationModel":
        """Fit the model using train and validation data."""

    def predict_proba(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Predict class probabilities for a prepared split."""

    def predict(
        self,
        data: Any,
        split: str = "test",
        threshold: float | None = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Predict labels for a prepared split."""

    def get_training_history(self) -> dict[str, Any]:
        """Return the training history."""

    def save(self, path: Path) -> Path:
        """Persist model parameters."""

    def load(self, path: Path) -> "ClassificationModel":
        """Restore model parameters."""

    def count_parameters(self) -> int:
        """Return trainable parameter count when meaningful."""


class RegressionModel(Protocol):
    """Common regression interface expected by the experiment runner."""

    model_name: str
    history: dict[str, Any]

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "RegressionModel":
        """Fit the model using train and validation data."""

    def predict(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Predict a continuous target in its original units."""

    def get_training_history(self) -> dict[str, Any]:
        """Return the training history."""

    def save(self, path: Path) -> Path:
        """Persist model parameters."""

    def load(self, path: Path) -> "RegressionModel":
        """Restore model parameters."""

    def count_parameters(self) -> int:
        """Return trainable parameter count when meaningful."""


def create_model(
    model_name: str,
    task: str,
    data_metadata: Any,
    model_config: dict[str, Any] | None = None,
) -> ClassificationModel | RegressionModel:
    """Create a model wrapper with a task-appropriate shared interface."""

    if task not in {"classification", "regression"}:
        raise ValueError(
            f"Unsupported task {task!r}; expected classification or regression."
        )

    metadata = _as_metadata_dict(data_metadata)
    config = dict(model_config or {})
    normalized = model_name.lower().replace("-", "_")

    if task == "classification":
        if normalized in {"baseline", "baseline_logistic", "logistic_regression"}:
            return LogisticRegressionBaseline(metadata, config)
        if normalized in {"tabnet", "tabnetclassifier", "tabnet_classifier"}:
            return TabNetWrapper(metadata, config)
        if normalized in {"tabtransformer", "tab_transformer"}:
            network = _build_tab_transformer(metadata, config)
            return TorchClassifierWrapper(
                "tab_transformer", network, metadata, config
            )
        if normalized in {"fttransformer", "ft_transformer"}:
            network = _build_ft_transformer(metadata, config)
            return TorchClassifierWrapper("ft_transformer", network, metadata, config)
        if normalized in {"saint", "saint_supervised"}:
            network = _build_saint(metadata, config)
            return TorchClassifierWrapper(
                "saint_supervised", network, metadata, config
            )
    else:
        if normalized in {"baseline", "baseline_ridge", "ridge"}:
            return RidgeRegressionBaseline(metadata, config)
        if normalized in {"tabnet", "tabnetregressor", "tabnet_regressor"}:
            return TabNetRegressorWrapper(metadata, config)
        if normalized in {"tabtransformer", "tab_transformer"}:
            network = _build_tab_transformer(metadata, config)
            return TorchRegressorWrapper(
                "tab_transformer", network, metadata, config
            )
        if normalized in {"fttransformer", "ft_transformer"}:
            network = _build_ft_transformer(metadata, config)
            return TorchRegressorWrapper("ft_transformer", network, metadata, config)
        if normalized in {"saint", "saint_supervised"}:
            network = _build_saint(metadata, config)
            return TorchRegressorWrapper(
                "saint_supervised", network, metadata, config
            )

    raise ValueError(f"Unknown model_name: {model_name!r}")


def optional_dependency_report() -> dict[str, bool]:
    """Report optional deep-learning dependencies used by this notebook."""

    return {
        "torch": TORCH_AVAILABLE,
        "pytorch_tabnet": TABNET_AVAILABLE,
    }


class LogisticRegressionBaseline:
    """Simple classical baseline with one-hot categorical features."""

    def __init__(self, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        self.model_name = "baseline_logistic"
        self.metadata = metadata
        self.config = config
        self.history: dict[str, Any] = {}
        self.encoder: OneHotEncoder | None = None
        self.classifier = LogisticRegression(
            max_iter=int(config.get("max_iter", 1000)),
            C=float(config.get("C", 1.0)),
            class_weight=config.get("class_weight", None),
            solver=str(config.get("solver", "lbfgs")),
            n_jobs=config.get("n_jobs", None),
            random_state=config.get("seed", None),
        )

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "LogisticRegressionBaseline":
        X_train = self._features(data, "train", fit=True)
        X_valid = self._features(data, "valid", fit=False)
        self.classifier.fit(X_train, data.y_train)

        train_proba = self.classifier.predict_proba(X_train)
        valid_proba = self.classifier.predict_proba(X_valid)
        labels = list(range(len(self.metadata["class_names"])))
        self.history = {
            "train_log_loss": [
                float(log_loss(data.y_train, train_proba, labels=labels))
            ],
            "valid_log_loss": [
                float(log_loss(data.y_valid, valid_proba, labels=labels))
            ],
            "best_epoch": 1,
            "epochs_trained": 1,
            "best_metric": "valid_log_loss",
            "estimator": "sklearn.linear_model.LogisticRegression",
        }
        self.save(checkpoint_path)
        return self

    def predict_proba(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del batch_size
        X = self._features(data, split, fit=False)
        return self.classifier.predict_proba(X).astype(np.float64)

    def predict(
        self,
        data: Any,
        split: str = "test",
        threshold: float | None = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        proba = self.predict_proba(data, split=split, batch_size=batch_size)
        if proba.shape[1] == 2 and threshold is not None:
            return (proba[:, 1] >= threshold).astype(np.int64)
        return np.argmax(proba, axis=1).astype(np.int64)

    def get_training_history(self) -> dict[str, Any]:
        return self.history

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": self.classifier,
                "encoder": self.encoder,
                "metadata": self.metadata,
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        return path

    def load(self, path: Path) -> "LogisticRegressionBaseline":
        payload = joblib.load(path)
        self.classifier = payload["classifier"]
        self.encoder = payload["encoder"]
        self.metadata = payload["metadata"]
        self.config = payload["config"]
        self.history = payload.get("history", {})
        return self

    def count_parameters(self) -> int:
        if not hasattr(self.classifier, "coef_"):
            return 0
        return int(self.classifier.coef_.size + self.classifier.intercept_.size)

    def get_embedding(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del data, split, batch_size
        raise NotImplementedError("The logistic baseline is not a latent encoder.")

    def _features(
        self,
        data: Any,
        split: str,
        fit: bool,
    ) -> sparse.spmatrix | np.ndarray:
        features, self.encoder = _classical_features(
            data=data,
            split=split,
            encoder=self.encoder,
            fit=fit,
            cardinalities=tuple(self.metadata["categorical_cardinalities"]),
        )
        return features


class RidgeRegressionBaseline:
    """Regularized linear baseline over the shared prepared features."""

    def __init__(self, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        self.model_name = "baseline_ridge"
        self.metadata = metadata
        self.config = config
        self.history: dict[str, Any] = {}
        self.encoder: OneHotEncoder | None = None
        self.regressor = Ridge(
            alpha=float(config.get("alpha", 1.0)),
            fit_intercept=bool(config.get("fit_intercept", True)),
            solver=str(config.get("solver", "auto")),
            max_iter=config.get("max_iter", None),
            tol=float(config.get("tol", 1e-4)),
            random_state=config.get("seed", None),
        )

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "RidgeRegressionBaseline":
        del training_config
        X_train = self._features(data, "train", fit=True)
        X_valid = self._features(data, "valid", fit=False)
        self.regressor.fit(X_train, data.y_train_scaled)
        train_prediction = data.inverse_transform_target(
            self.regressor.predict(X_train)
        )
        valid_prediction = data.inverse_transform_target(
            self.regressor.predict(X_valid)
        )
        self.history = {
            "train_rmse": [
                float(np.sqrt(mean_squared_error(data.y_train, train_prediction)))
            ],
            "valid_rmse": [
                float(np.sqrt(mean_squared_error(data.y_valid, valid_prediction)))
            ],
            "best_epoch": 1,
            "epochs_trained": 1,
            "best_metric": "valid_rmse",
            "estimator": "sklearn.linear_model.Ridge",
        }
        self.save(checkpoint_path)
        return self

    def predict(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del batch_size
        X = self._features(data, split, fit=False)
        scaled_prediction = self.regressor.predict(X)
        return data.inverse_transform_target(scaled_prediction).reshape(-1)

    def get_training_history(self) -> dict[str, Any]:
        return self.history

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "regressor": self.regressor,
                "encoder": self.encoder,
                "metadata": self.metadata,
                "config": self.config,
                "history": self.history,
            },
            path,
        )
        return path

    def load(self, path: Path) -> "RidgeRegressionBaseline":
        payload = joblib.load(path)
        self.regressor = payload["regressor"]
        self.encoder = payload["encoder"]
        self.metadata = payload["metadata"]
        self.config = payload["config"]
        self.history = payload.get("history", {})
        return self

    def count_parameters(self) -> int:
        if not hasattr(self.regressor, "coef_"):
            return 0
        intercept = np.asarray(self.regressor.intercept_)
        return int(np.asarray(self.regressor.coef_).size + intercept.size)

    def get_embedding(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del data, split, batch_size
        raise NotImplementedError("The Ridge baseline is not a latent encoder.")

    def _features(
        self,
        data: Any,
        split: str,
        fit: bool,
    ) -> sparse.spmatrix | np.ndarray:
        features, self.encoder = _classical_features(
            data=data,
            split=split,
            encoder=self.encoder,
            fit=fit,
            cardinalities=tuple(self.metadata["categorical_cardinalities"]),
        )
        return features


class TabNetWrapper:
    """Wrapper around pytorch-tabnet.TabNetClassifier."""

    def __init__(self, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        _require_tabnet()
        self.model_name = "tabnet"
        self.metadata = metadata
        self.config = config
        self.history: dict[str, Any] = {}
        self.model = self._make_model(config)

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "TabNetWrapper":
        eval_metric = self.config.get(
            "eval_metric",
            ["auc"] if len(self.metadata["class_names"]) == 2 else ["accuracy"],
        )
        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        self.model.fit(
            X_train=data.X_train.astype(np.float32),
            y_train=data.y_train,
            eval_set=[(data.X_valid.astype(np.float32), data.y_valid)],
            eval_name=["valid"],
            eval_metric=eval_metric,
            max_epochs=int(training_config["max_epochs"]),
            patience=int(training_config["patience"]),
            batch_size=int(training_config["batch_size"]),
            virtual_batch_size=int(
                self.config.get(
                    "virtual_batch_size",
                    max(32, min(256, int(training_config["batch_size"]) // 4)),
                )
            ),
            num_workers=int(training_config.get("num_workers", 0)),
            drop_last=False,
        )
        self.history = _tabnet_history_to_dict(
            self.model,
            estimator="pytorch_tabnet.TabNetClassifier",
        )
        epochs_trained = len(self.history.get("loss", []))
        self.history["epochs_trained"] = epochs_trained
        self.history["reached_epoch_budget"] = (
            epochs_trained >= int(training_config["max_epochs"])
        )
        self.save(checkpoint_path)
        return self

    def predict_proba(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del batch_size
        _, _, X = _split_arrays(data, split)
        return self.model.predict_proba(X.astype(np.float32)).astype(np.float64)

    def predict(
        self,
        data: Any,
        split: str = "test",
        threshold: float | None = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        proba = self.predict_proba(data, split=split, batch_size=batch_size)
        if proba.shape[1] == 2 and threshold is not None:
            return (proba[:, 1] >= threshold).astype(np.int64)
        return np.argmax(proba, axis=1).astype(np.int64)

    def get_training_history(self) -> dict[str, Any]:
        return self.history

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base_path = path.with_suffix("") if path.suffix == ".zip" else path
        saved_path = self.model.save_model(str(base_path))
        return Path(saved_path)

    def load(self, path: Path) -> "TabNetWrapper":
        path = Path(path)
        load_path = path if path.exists() else path.with_suffix(".zip")
        self.model.load_model(str(load_path))
        return self

    def count_parameters(self) -> int:
        network = getattr(self.model, "network", None)
        if network is None:
            return 0
        return int(sum(parameter.numel() for parameter in network.parameters()))

    def get_embedding(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del data, split, batch_size
        raise NotImplementedError(
            "TabNet exposes masks and explanations, but this wrapper does not "
            "treat them as a stable latent embedding."
        )

    def _make_model(self, config: dict[str, Any]) -> Any:
        return TabNetClassifier(
            cat_idxs=list(self.metadata["categorical_indices"]),
            cat_dims=list(self.metadata["categorical_cardinalities"]),
            cat_emb_dim=int(config.get("cat_emb_dim", 8)),
            n_d=int(config.get("n_d", 24)),
            n_a=int(config.get("n_a", 24)),
            n_steps=int(config.get("n_steps", 4)),
            gamma=float(config.get("gamma", 1.4)),
            lambda_sparse=float(config.get("lambda_sparse", 1e-4)),
            optimizer_params={
                "lr": float(config.get("learning_rate", 1e-3)),
                "weight_decay": float(config.get("weight_decay", 0.0)),
            },
            seed=int(config.get("seed", 42)),
            verbose=int(config.get("verbose", 0)),
            device_name=str(config.get("device", "auto")),
        )


class TabNetRegressorWrapper:
    """Wrapper around pytorch-tabnet.TabNetRegressor."""

    def __init__(self, metadata: dict[str, Any], config: dict[str, Any]) -> None:
        _require_tabnet()
        self.model_name = "tabnet"
        self.metadata = metadata
        self.config = config
        self.history: dict[str, Any] = {}
        self.model = self._make_model(config)

    def fit(
        self,
        data: Any,
        training_config: dict[str, Any],
        checkpoint_path: Path,
    ) -> "TabNetRegressorWrapper":
        eval_metric = self.config.get("eval_metric", ["rmse"])
        if isinstance(eval_metric, str):
            eval_metric = [eval_metric]
        self.model.fit(
            X_train=data.X_train.astype(np.float32),
            y_train=data.y_train_scaled.reshape(-1, 1),
            eval_set=[
                (
                    data.X_valid.astype(np.float32),
                    data.y_valid_scaled.reshape(-1, 1),
                )
            ],
            eval_name=["valid"],
            eval_metric=list(eval_metric),
            max_epochs=int(training_config["max_epochs"]),
            patience=int(training_config["patience"]),
            batch_size=int(training_config["batch_size"]),
            virtual_batch_size=int(
                self.config.get(
                    "virtual_batch_size",
                    max(32, min(256, int(training_config["batch_size"]) // 4)),
                )
            ),
            num_workers=int(training_config.get("num_workers", 0)),
            drop_last=False,
        )
        self.history = _tabnet_history_to_dict(
            self.model,
            estimator="pytorch_tabnet.TabNetRegressor",
        )
        self.history = _tabnet_regression_history_in_original_units(
            self.history,
            target_std=float(data.preprocessing_state.target_std),
        )
        epochs_trained = len(self.history.get("loss", []))
        self.history["epochs_trained"] = epochs_trained
        self.history["reached_epoch_budget"] = (
            epochs_trained >= int(training_config["max_epochs"])
        )
        self.save(checkpoint_path)
        return self

    def predict(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del batch_size
        _, _, X = _split_arrays(data, split)
        scaled_prediction = self.model.predict(X.astype(np.float32)).reshape(-1)
        return data.inverse_transform_target(scaled_prediction).reshape(-1)

    def get_training_history(self) -> dict[str, Any]:
        return self.history

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        base_path = path.with_suffix("") if path.suffix == ".zip" else path
        saved_path = self.model.save_model(str(base_path))
        return Path(saved_path)

    def load(self, path: Path) -> "TabNetRegressorWrapper":
        path = Path(path)
        load_path = path if path.exists() else path.with_suffix(".zip")
        self.model.load_model(str(load_path))
        return self

    def count_parameters(self) -> int:
        network = getattr(self.model, "network", None)
        if network is None:
            return 0
        return int(sum(parameter.numel() for parameter in network.parameters()))

    def get_embedding(
        self,
        data: Any,
        split: str = "test",
        batch_size: int = 4096,
    ) -> np.ndarray:
        del data, split, batch_size
        raise NotImplementedError(
            "This wrapper does not expose a stable TabNet latent representation."
        )

    def _make_model(self, config: dict[str, Any]) -> Any:
        return TabNetRegressor(
            cat_idxs=list(self.metadata["categorical_indices"]),
            cat_dims=list(self.metadata["categorical_cardinalities"]),
            cat_emb_dim=int(config.get("cat_emb_dim", 8)),
            n_d=int(config.get("n_d", 24)),
            n_a=int(config.get("n_a", 24)),
            n_steps=int(config.get("n_steps", 4)),
            gamma=float(config.get("gamma", 1.4)),
            lambda_sparse=float(config.get("lambda_sparse", 1e-4)),
            optimizer_params={
                "lr": float(config.get("learning_rate", 1e-3)),
                "weight_decay": float(config.get("weight_decay", 0.0)),
            },
            seed=int(config.get("seed", 42)),
            verbose=int(config.get("verbose", 0)),
            device_name=str(config.get("device", "auto")),
        )


if TORCH_AVAILABLE:

    class TabTransformerNetwork(nn.Module):
        """TabTransformer-style categorical contextualization network."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            n_classes: int,
            d_token: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
            mlp_hidden: tuple[int, ...],
        ) -> None:
            super().__init__()
            self.n_num_features = n_num_features
            self.n_cat_features = len(cat_cardinalities)
            self.cat_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, d_token)
                    for cardinality in cat_cardinalities
                ]
            )
            self.cat_column_embeddings = nn.Parameter(
                torch.zeros(self.n_cat_features, d_token)
            )
            if self.n_cat_features > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_token,
                    nhead=n_heads,
                    dim_feedforward=4 * d_token,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer,
                    num_layers=n_layers,
                    enable_nested_tensor=False,
                )
            else:
                self.transformer = None
            head_dim = self.n_cat_features * d_token + n_num_features
            self.head = _build_mlp(head_dim, mlp_hidden, n_classes, dropout)
            self._reset_parameters()

        def _reset_parameters(self) -> None:
            nn.init.normal_(self.cat_column_embeddings, std=0.02)

        def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            pieces: list[torch.Tensor] = []
            if self.n_cat_features > 0:
                cat_tokens = torch.stack(
                    [
                        embedding(x_cat[:, idx])
                        for idx, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
                cat_tokens = cat_tokens + self.cat_column_embeddings.unsqueeze(0)
                assert self.transformer is not None
                cat_context = self.transformer(cat_tokens).flatten(start_dim=1)
                pieces.append(cat_context)
            if self.n_num_features > 0:
                pieces.append(x_num)
            return torch.cat(pieces, dim=1)

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x_cat, x_num))


    class TabularFeatureTokenizer(nn.Module):
        """Tokenize categorical and numerical columns into transformer tokens."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            d_token: int,
            add_cls: bool,
        ) -> None:
            super().__init__()
            self.n_cat_features = len(cat_cardinalities)
            self.n_num_features = n_num_features
            self.add_cls = add_cls
            self.cat_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, d_token)
                    for cardinality in cat_cardinalities
                ]
            )
            self.cat_column_embeddings = nn.Parameter(
                torch.zeros(self.n_cat_features, d_token)
            )
            self.num_weight = nn.Parameter(torch.empty(n_num_features, d_token))
            self.num_bias = nn.Parameter(torch.empty(n_num_features, d_token))
            self.cls_token = (
                nn.Parameter(torch.empty(1, 1, d_token)) if add_cls else None
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            if self.n_cat_features > 0:
                nn.init.normal_(self.cat_column_embeddings, std=0.02)
            if self.n_num_features > 0:
                nn.init.xavier_uniform_(self.num_weight)
                nn.init.zeros_(self.num_bias)
            if self.cls_token is not None:
                nn.init.normal_(self.cls_token, std=0.02)

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            tokens: list[torch.Tensor] = []
            if self.n_cat_features > 0:
                cat_tokens = torch.stack(
                    [
                        embedding(x_cat[:, idx])
                        for idx, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
                tokens.append(cat_tokens + self.cat_column_embeddings.unsqueeze(0))
            if self.n_num_features > 0:
                num_tokens = (
                    x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)
                    + self.num_bias.unsqueeze(0)
                )
                tokens.append(num_tokens)
            if not tokens:
                raise ValueError("At least one tabular feature is required.")
            output = torch.cat(tokens, dim=1)
            if self.cls_token is not None:
                cls = self.cls_token.expand(output.shape[0], -1, -1)
                output = torch.cat([cls, output], dim=1)
            return output


    class FTTransformerNetwork(nn.Module):
        """Feature Tokenizer Transformer for mixed tabular inputs."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            n_classes: int,
            d_token: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.tokenizer = TabularFeatureTokenizer(
                cat_cardinalities,
                n_num_features,
                d_token,
                add_cls=True,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(d_token),
                nn.ReLU(),
                nn.Linear(d_token, n_classes),
            )

        def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            tokens = self.tokenizer(x_cat, x_num)
            return self.transformer(tokens)[:, 0]

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x_cat, x_num))


    class LegacySaintNetwork(nn.Module):
        """Legacy SAINT-style network retained to restore existing checkpoints."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            n_classes: int,
            d_token: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
            use_row_attention: bool,
            mlp_hidden: tuple[int, ...],
        ) -> None:
            super().__init__()
            self.tokenizer = TabularFeatureTokenizer(
                cat_cardinalities,
                n_num_features,
                d_token,
                add_cls=False,
            )
            self.use_row_attention = use_row_attention
            self.column_blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_token,
                        nhead=n_heads,
                        dim_feedforward=4 * d_token,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.row_blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_token,
                        nhead=n_heads,
                        dim_feedforward=4 * d_token,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.head = _build_mlp(d_token, mlp_hidden, n_classes, dropout)

        def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            tokens = self.tokenizer(x_cat, x_num)
            for column_block, row_block in zip(self.column_blocks, self.row_blocks):
                tokens = column_block(tokens)
                if self.use_row_attention and tokens.shape[0] > 1:
                    row_tokens = row_block(tokens.transpose(0, 1)).transpose(0, 1)
                    tokens = tokens + row_tokens
            return tokens.mean(dim=1)

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x_cat, x_num))


    class SaintFeatureTokenizer(nn.Module):
        """Embed a CLS token plus categorical and numerical SAINT features."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            d_token: int,
            numerical_hidden: int,
        ) -> None:
            super().__init__()
            self.n_cat_features = len(cat_cardinalities)
            self.n_num_features = n_num_features
            self.cat_embeddings = nn.ModuleList(
                [
                    nn.Embedding(cardinality, d_token)
                    for cardinality in cat_cardinalities
                ]
            )
            self.num_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(1, numerical_hidden),
                        nn.ReLU(),
                        nn.Linear(numerical_hidden, d_token),
                    )
                    for _ in range(n_num_features)
                ]
            )
            self.cls_token = nn.Parameter(torch.empty(1, 1, d_token))
            self.column_embeddings = nn.Parameter(
                torch.empty(1 + self.n_cat_features + n_num_features, d_token)
            )
            self.reset_parameters()

        @property
        def n_tokens(self) -> int:
            return 1 + self.n_cat_features + self.n_num_features

        def reset_parameters(self) -> None:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.column_embeddings, std=0.02)

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            batch_size = x_cat.shape[0] if x_cat.ndim == 2 else x_num.shape[0]
            tokens: list[torch.Tensor] = [
                self.cls_token.expand(batch_size, -1, -1)
            ]
            if self.n_cat_features > 0:
                tokens.append(
                    torch.stack(
                        [
                            embedding(x_cat[:, idx])
                            for idx, embedding in enumerate(self.cat_embeddings)
                        ],
                        dim=1,
                    )
                )
            if self.n_num_features > 0:
                tokens.append(
                    torch.stack(
                        [
                            embedding(x_num[:, idx : idx + 1])
                            for idx, embedding in enumerate(self.num_embeddings)
                        ],
                        dim=1,
                    )
                )
            output = torch.cat(tokens, dim=1)
            return output + self.column_embeddings.unsqueeze(0)


    class SaintIntersampleBlock(nn.Module):
        """Apply attention across rows using each complete row as one token."""

        def __init__(
            self,
            n_tokens: int,
            d_token: int,
            n_heads: int,
            dim_head: int,
            dropout: float,
            ff_multiplier: float,
        ) -> None:
            super().__init__()
            self.n_tokens = n_tokens
            self.d_token = d_token
            self.n_heads = n_heads
            self.dim_head = dim_head
            row_dim = n_tokens * d_token
            inner_dim = n_heads * dim_head
            hidden_dim = max(row_dim, int(round(row_dim * ff_multiplier)))

            self.attention_norm = nn.LayerNorm(row_dim)
            self.to_qkv = nn.Linear(row_dim, 3 * inner_dim, bias=False)
            self.to_out = nn.Linear(inner_dim, row_dim)
            self.attention_dropout = nn.Dropout(dropout)
            self.feedforward_norm = nn.LayerNorm(row_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(row_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, row_dim),
                nn.Dropout(dropout),
            )

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            batch_size = tokens.shape[0]
            rows = tokens.reshape(batch_size, -1).unsqueeze(0)
            normalized = self.attention_norm(rows)
            query, key, value = self.to_qkv(normalized).chunk(3, dim=-1)

            def split_heads(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape(
                    1, batch_size, self.n_heads, self.dim_head
                ).transpose(1, 2)

            query = split_heads(query)
            key = split_heads(key)
            value = split_heads(value)
            scale = self.dim_head**-0.5
            attention = torch.matmul(query, key.transpose(-2, -1)) * scale
            attention = self.attention_dropout(torch.softmax(attention, dim=-1))
            attended = torch.matmul(attention, value)
            attended = attended.transpose(1, 2).reshape(1, batch_size, -1)
            rows = rows + self.to_out(attended)
            rows = rows + self.feedforward(self.feedforward_norm(rows))
            return rows.squeeze(0).reshape(batch_size, self.n_tokens, self.d_token)


    class SaintNetwork(nn.Module):
        """Supervised SAINT with column and optional intersample attention."""

        def __init__(
            self,
            cat_cardinalities: tuple[int, ...],
            n_num_features: int,
            n_classes: int,
            d_token: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
            use_row_attention: bool,
            mlp_hidden: tuple[int, ...],
            numerical_hidden: int,
            row_attention_dim_head: int,
            row_ff_multiplier: float,
        ) -> None:
            super().__init__()
            self.tokenizer = SaintFeatureTokenizer(
                cat_cardinalities=cat_cardinalities,
                n_num_features=n_num_features,
                d_token=d_token,
                numerical_hidden=numerical_hidden,
            )
            self.use_row_attention = use_row_attention
            self.column_blocks = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_token,
                        nhead=n_heads,
                        dim_feedforward=4 * d_token,
                        dropout=dropout,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.row_blocks = nn.ModuleList(
                [
                    SaintIntersampleBlock(
                        n_tokens=self.tokenizer.n_tokens,
                        d_token=d_token,
                        n_heads=n_heads,
                        dim_head=row_attention_dim_head,
                        dropout=dropout,
                        ff_multiplier=row_ff_multiplier,
                    )
                    for _ in range(n_layers)
                ]
                if use_row_attention
                else []
            )
            self.head = _build_mlp(d_token, mlp_hidden, n_classes, dropout)

        def encode(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            tokens = self.tokenizer(x_cat, x_num)
            for layer_idx, column_block in enumerate(self.column_blocks):
                tokens = column_block(tokens)
                if self.use_row_attention:
                    tokens = self.row_blocks[layer_idx](tokens)
            return tokens[:, 0]

        def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
            return self.head(self.encode(x_cat, x_num))


    class _TorchModelWrapper:
        """Shared persistence and inference operations for PyTorch wrappers."""

        def __init__(
            self,
            model_name: str,
            network: nn.Module,
            metadata: dict[str, Any],
            config: dict[str, Any],
        ) -> None:
            self.model_name = model_name
            self.network = network
            self.metadata = metadata
            self.config = config
            self.device = torch.device(str(config.get("device", "cpu")))
            self.network.to(self.device)
            self.history: dict[str, Any] = {}

        def get_training_history(self) -> dict[str, Any]:
            return self.history

        def save(self, path: Path) -> Path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": self.network.state_dict(),
                    "metadata": self.metadata,
                    "config": self.config,
                    "history": self.history,
                    "model_name": self.model_name,
                },
                path,
            )
            return path

        def load(self, path: Path) -> "_TorchModelWrapper":
            payload = torch.load(path, map_location=self.device)
            checkpoint_model = payload.get("model_name")
            if checkpoint_model and checkpoint_model != self.model_name:
                raise ValueError(
                    f"Checkpoint contains {checkpoint_model!r}, not "
                    f"{self.model_name!r}."
                )
            self.network.load_state_dict(payload["state_dict"])
            self.history = payload.get("history", {})
            return self

        def count_parameters(self) -> int:
            return int(
                sum(
                    parameter.numel()
                    for parameter in self.network.parameters()
                    if parameter.requires_grad
                )
            )

        def get_embedding(
            self,
            data: Any,
            split: str = "test",
            batch_size: int = 4096,
        ) -> np.ndarray:
            self._validate_inference_batch_size(batch_size)
            X_cat, X_num, _ = _split_arrays(data, split)
            dataset = _torch_dataset(X_cat, X_num, np.zeros(len(X_cat), dtype=np.int64))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            embeddings: list[np.ndarray] = []
            self.network.eval()
            with torch.no_grad():
                for x_cat, x_num, _ in loader:
                    x_cat = x_cat.to(self.device)
                    x_num = x_num.to(self.device)
                    embedding = self.network.encode(x_cat, x_num)
                    embeddings.append(embedding.detach().cpu().numpy())
            return np.concatenate(embeddings, axis=0).astype(np.float32)

        def _validate_inference_batch_size(self, batch_size: int) -> None:
            if self.model_name != "saint_supervised":
                return
            if self.config.get("implementation_version") != "saint_colrow_v1":
                return
            if not bool(self.config.get("use_row_attention", True)):
                return
            expected = int(self.config.get("inference_batch_size", batch_size))
            if batch_size != expected:
                raise ValueError(
                    "SAINT intersample attention is batch-dependent; use the "
                    f"configured inference_batch_size={expected}, not {batch_size}."
                )

        def _predict_outputs(
            self,
            data: Any,
            split: str,
            batch_size: int,
        ) -> np.ndarray:
            X_cat, X_num, _ = _split_arrays(data, split)
            labels = np.zeros(len(X_cat), dtype=np.int64)
            dataset = _torch_dataset(X_cat, X_num, labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            outputs: list[np.ndarray] = []
            self.network.eval()
            with torch.no_grad():
                for x_cat, x_num, _ in loader:
                    x_cat = x_cat.to(self.device)
                    x_num = x_num.to(self.device)
                    batch_outputs = self.network(x_cat, x_num)
                    outputs.append(batch_outputs.detach().cpu().numpy())
            return np.concatenate(outputs, axis=0).astype(np.float32)


    class TorchClassifierWrapper(_TorchModelWrapper):
        """Task-specific interface around a PyTorch classifier."""

        def fit(
            self,
            data: Any,
            training_config: dict[str, Any],
            checkpoint_path: Path,
        ) -> "TorchClassifierWrapper":
            from .training import train_torch_classifier

            self._store_batch_configuration(training_config)
            self.history = train_torch_classifier(
                model=self,
                data=data,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
            )
            return self

        def predict_proba(
            self,
            data: Any,
            split: str = "test",
            batch_size: int = 4096,
        ) -> np.ndarray:
            self._validate_inference_batch_size(batch_size)
            logits = self._predict_outputs(data, split=split, batch_size=batch_size)
            probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
            return probabilities.astype(np.float64)

        def predict(
            self,
            data: Any,
            split: str = "test",
            threshold: float | None = None,
            batch_size: int = 4096,
        ) -> np.ndarray:
            proba = self.predict_proba(data, split=split, batch_size=batch_size)
            if proba.shape[1] == 2 and threshold is not None:
                return (proba[:, 1] >= threshold).astype(np.int64)
            return np.argmax(proba, axis=1).astype(np.int64)

        def _store_batch_configuration(
            self,
            training_config: dict[str, Any],
        ) -> None:
            if self.model_name == "saint_supervised":
                self.config["batch_size"] = int(training_config["batch_size"])
                self.config["inference_batch_size"] = int(
                    training_config["inference_batch_size"]
                )


    class TorchRegressorWrapper(_TorchModelWrapper):
        """Task-specific interface around a scalar PyTorch regressor."""

        def fit(
            self,
            data: Any,
            training_config: dict[str, Any],
            checkpoint_path: Path,
        ) -> "TorchRegressorWrapper":
            from .training import train_torch_regressor

            if self.model_name == "saint_supervised":
                self.config["batch_size"] = int(training_config["batch_size"])
                self.config["inference_batch_size"] = int(
                    training_config["inference_batch_size"]
                )
            self.history = train_torch_regressor(
                model=self,
                data=data,
                training_config=training_config,
                checkpoint_path=checkpoint_path,
            )
            return self

        def predict(
            self,
            data: Any,
            split: str = "test",
            batch_size: int = 4096,
        ) -> np.ndarray:
            self._validate_inference_batch_size(batch_size)
            scaled = self._predict_outputs(
                data,
                split=split,
                batch_size=batch_size,
            )
            if scaled.ndim != 2 or scaled.shape[1] != 1:
                raise ValueError("A regression network must return shape (n_rows, 1).")
            return data.inverse_transform_target(scaled[:, 0]).reshape(-1)


else:

    class TorchClassifierWrapper:  # type: ignore[no-redef]
        """Placeholder that raises a clear dependency error."""

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class TorchRegressorWrapper:  # type: ignore[no-redef]
        """Placeholder that raises a clear dependency error."""

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def _build_tab_transformer(
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> Any:
    _require_torch()
    return TabTransformerNetwork(
        cat_cardinalities=tuple(metadata["categorical_cardinalities"]),
        n_num_features=int(metadata["n_numerical_features"]),
        n_classes=_output_dimension(metadata),
        d_token=int(config.get("d_token", 32)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.10)),
        mlp_hidden=tuple(config.get("mlp_hidden", (128, 64))),
    )


def _build_ft_transformer(metadata: dict[str, Any], config: dict[str, Any]) -> Any:
    _require_torch()
    return FTTransformerNetwork(
        cat_cardinalities=tuple(metadata["categorical_cardinalities"]),
        n_num_features=int(metadata["n_numerical_features"]),
        n_classes=_output_dimension(metadata),
        d_token=int(config.get("d_token", 32)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.10)),
    )


def _build_saint(metadata: dict[str, Any], config: dict[str, Any]) -> Any:
    _require_torch()
    implementation_version = str(
        config.get("implementation_version", "saint_column_v1")
    )
    if implementation_version == "saint_legacy_columnwise_v0":
        return LegacySaintNetwork(
            cat_cardinalities=tuple(metadata["categorical_cardinalities"]),
            n_num_features=int(metadata["n_numerical_features"]),
            n_classes=_output_dimension(metadata),
            d_token=int(config.get("d_token", 32)),
            n_heads=int(config.get("n_heads", 4)),
            n_layers=int(config.get("n_layers", 2)),
            dropout=float(config.get("dropout", 0.10)),
            use_row_attention=bool(config.get("use_row_attention", True)),
            mlp_hidden=tuple(config.get("mlp_hidden", (128, 64))),
        )
    if implementation_version not in {"saint_column_v1", "saint_colrow_v1"}:
        raise ValueError(
            f"Unsupported SAINT implementation_version: {implementation_version!r}"
        )
    use_row_attention = bool(config.get("use_row_attention", False))
    if implementation_version == "saint_column_v1" and use_row_attention:
        raise ValueError(
            "saint_column_v1 requires use_row_attention=False. Use "
            "saint_colrow_v1 for an explicitly batch-contextual SAINT variant."
        )
    return SaintNetwork(
        cat_cardinalities=tuple(metadata["categorical_cardinalities"]),
        n_num_features=int(metadata["n_numerical_features"]),
        n_classes=_output_dimension(metadata),
        d_token=int(config.get("d_token", 32)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 2)),
        dropout=float(config.get("dropout", 0.10)),
        use_row_attention=use_row_attention,
        mlp_hidden=tuple(config.get("mlp_hidden", (128, 64))),
        numerical_hidden=int(config.get("numerical_embedding_hidden", 32)),
        row_attention_dim_head=int(config.get("row_attention_dim_head", 32)),
        row_ff_multiplier=float(config.get("row_ff_multiplier", 2.0)),
    )


def _build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    dropout: float,
) -> Any:
    layers: list[Any] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


def _torch_dataset(X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray) -> Any:
    _require_torch()
    return TensorDataset(
        torch.as_tensor(X_cat, dtype=torch.long),
        torch.as_tensor(X_num, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )


def _split_arrays(data: Any, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split == "train":
        return data.X_cat_train, data.X_num_train, data.X_train
    if split == "valid":
        return data.X_cat_valid, data.X_num_valid, data.X_valid
    if split == "test":
        return data.X_cat_test, data.X_num_test, data.X_test
    raise ValueError(f"Unknown split {split!r}; expected train, valid, or test.")


def _classical_features(
    data: Any,
    split: str,
    encoder: OneHotEncoder | None,
    fit: bool,
    cardinalities: tuple[int, ...],
) -> tuple[sparse.spmatrix | np.ndarray, OneHotEncoder | None]:
    """Build baseline features without fitting category state outside train."""

    X_cat, X_num, _ = _split_arrays(data, split)
    if X_cat.shape[1] == 0:
        return X_num, encoder

    if fit:
        categories = [
            np.arange(cardinality, dtype=np.int64)
            for cardinality in cardinalities
        ]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
        cat_features = encoder.fit_transform(X_cat)
    else:
        if encoder is None:
            raise RuntimeError("Baseline encoder is not fitted.")
        cat_features = encoder.transform(X_cat)

    if X_num.shape[1] == 0:
        return cat_features, encoder
    features = sparse.hstack(
        [cat_features, sparse.csr_matrix(X_num.astype(np.float32))],
        format="csr",
    )
    return features, encoder


def _as_metadata_dict(data_metadata: Any) -> dict[str, Any]:
    if hasattr(data_metadata, "metadata"):
        return dict(data_metadata.metadata())
    return dict(data_metadata)


def _output_dimension(metadata: dict[str, Any]) -> int:
    if "n_classes" in metadata:
        return int(metadata["n_classes"])
    if "n_outputs" in metadata:
        return int(metadata["n_outputs"])
    raise ValueError("Model metadata must define n_classes or n_outputs.")


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for TabTransformer, FT-Transformer, and SAINT. "
            "Install torch before running the deep tabular models."
        )


def _require_tabnet() -> None:
    if not TABNET_AVAILABLE:
        raise ImportError(
            "pytorch-tabnet is required for TabNet models. Install pytorch-tabnet "
            "before running the TabNet experiment."
        )


def _tabnet_history_to_dict(model: Any, estimator: str) -> dict[str, Any]:
    raw_history = getattr(model, "history", {})
    if hasattr(raw_history, "history"):
        raw_history = raw_history.history
    if not isinstance(raw_history, dict):
        raw_history = dict(raw_history)
    history = {
        key: [float(value) for value in values]
        if isinstance(values, (list, tuple))
        else values
        for key, values in raw_history.items()
    }
    best_epoch = getattr(model, "best_epoch", None)
    if best_epoch is not None:
        history["best_epoch"] = int(best_epoch) + 1
    history["estimator"] = estimator
    return history


def _tabnet_regression_history_in_original_units(
    history: dict[str, Any],
    target_std: float,
) -> dict[str, Any]:
    """Convert TabNet RMSE histories from standardized to original target units."""

    scale = abs(float(target_std))
    converted = dict(history)
    for key, values in history.items():
        if key.lower().endswith("rmse") and isinstance(values, list):
            converted[key] = [scale * float(value) for value in values]
    converted["rmse_history_units"] = "original_target_units"
    return converted
