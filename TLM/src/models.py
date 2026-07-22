"""Model wrappers for the TLM multiclass tabular benchmark."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import joblib
import numpy as np
import torch
from numpy.typing import NDArray
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ModelMetadata:
    """Dimensions and class information required to instantiate a model."""

    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_cardinalities: tuple[int, ...]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.class_names:
            raise ValueError("At least one class is required")
        if len(self.categorical_columns) != len(
            self.categorical_cardinalities
        ):
            raise ValueError("Categorical columns and cardinalities differ")
        if any(cardinality < 2 for cardinality in self.categorical_cardinalities):
            raise ValueError("Each categorical feature needs an unknown category")
        if not self.numerical_columns and not self.categorical_columns:
            raise ValueError("A model requires at least one input feature")

    @property
    def n_num(self) -> int:
        """Return the number of continuous inputs."""

        return len(self.numerical_columns)

    @property
    def n_cat(self) -> int:
        """Return the number of categorical inputs."""

        return len(self.categorical_columns)

    @property
    def n_classes(self) -> int:
        """Return the number of target classes."""

        return len(self.class_names)


@runtime_checkable
class ClassificationModel(Protocol):
    """Common surface consumed by the experiment runner."""

    model_name: str
    metadata: ModelMetadata
    history: list[dict[str, float]]
    best_epoch: int

    def fit(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        y: IntArray,
        *,
        X_num_valid: FloatArray | None,
        X_cat_valid: IntArray | None,
        y_valid: IntArray | None,
        class_weights: FloatArray,
        fit_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fit the model and return timing and selection information."""

    def predict_proba(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        """Return one probability vector per row."""

    def predict(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> IntArray:
        """Return one predicted class index per row."""

    def get_training_history(self) -> list[dict[str, float]]:
        """Return a copy of the epoch history."""

    def get_embedding(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        """Return the representation immediately before the classifier."""

    def parameter_count(self) -> int:
        """Return the number of fitted or trainable parameters."""

    def save(self, path: Path | str) -> Path:
        """Persist the fitted model."""

    def load(self, path: Path | str) -> None:
        """Restore a fitted model."""


class LogisticRegressionWrapper:
    """Linear baseline over normalized values and deterministic one-hot codes."""

    model_name = "logistic_regression"

    def __init__(
        self,
        metadata: ModelMetadata,
        model_config: Mapping[str, Any],
        device: str,
        seed: int,
    ) -> None:
        del device
        self.metadata = metadata
        self.model_config = dict(model_config)
        self.seed = int(seed)
        self.model = LogisticRegression(
            max_iter=int(self.model_config.get("max_iter", 2_000)),
            solver=str(self.model_config.get("solver", "lbfgs")),
            C=float(self.model_config.get("C", 1.0)),
            random_state=self.seed,
        )
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0

    def fit(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        y: IntArray,
        *,
        X_num_valid: FloatArray | None,
        X_cat_valid: IntArray | None,
        y_valid: IntArray | None,
        class_weights: FloatArray,
        fit_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Delegate fitting to the shared training module."""

        from src.training import fit_logistic_model

        return fit_logistic_model(
            self,
            X_num,
            X_cat,
            y,
            X_num_valid=X_num_valid,
            X_cat_valid=X_cat_valid,
            y_valid=y_valid,
            class_weights=class_weights,
            fit_config=fit_config,
        )

    def predict_proba(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        del batch_size
        values = self.model.predict_proba(self._design_matrix(X_num, X_cat))
        return np.asarray(values, dtype=np.float32)

    def predict(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> IntArray:
        return self.predict_proba(X_num, X_cat, batch_size).argmax(axis=1)

    def get_training_history(self) -> list[dict[str, float]]:
        return [dict(row) for row in self.history]

    def get_embedding(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        del batch_size
        return self._design_matrix(X_num, X_cat).astype(np.float32)

    def parameter_count(self) -> int:
        if not hasattr(self.model, "coef_"):
            return 0
        return int(self.model.coef_.size + self.model.intercept_.size)

    def save(self, path: Path | str) -> Path:
        destination = Path(path).with_suffix(".joblib")
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "metadata": asdict(self.metadata),
                "model_config": self.model_config,
                "seed": self.seed,
                "history": self.history,
                "best_epoch": self.best_epoch,
            },
            destination,
        )
        return destination

    def load(self, path: Path | str) -> None:
        payload = joblib.load(Path(path))
        if payload["metadata"] != asdict(self.metadata):
            raise ValueError("Checkpoint metadata does not match wrapper")
        if payload["model_config"] != self.model_config:
            raise ValueError("Checkpoint configuration does not match wrapper")
        self.model = payload["model"]
        self.history = [dict(row) for row in payload.get("history", [])]
        self.best_epoch = int(payload.get("best_epoch", 0))

    def _design_matrix(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
    ) -> FloatArray:
        blocks: list[FloatArray] = [np.asarray(X_num, dtype=np.float32)]
        for index, cardinality in enumerate(
            self.metadata.categorical_cardinalities
        ):
            values = np.asarray(X_cat[:, index], dtype=np.int64)
            if values.min(initial=0) < 0 or values.max(initial=0) >= cardinality:
                raise ValueError("Categorical value lies outside its cardinality")
            blocks.append(np.eye(cardinality, dtype=np.float32)[values])
        return np.concatenate(blocks, axis=1)


class TorchClassifierWrapper:
    """Thin adapter around a torch module with an ``encode`` method."""

    model_name = "torch_classifier"

    def __init__(
        self,
        metadata: ModelMetadata,
        model_config: Mapping[str, Any],
        device: str,
        seed: int,
        module: nn.Module,
    ) -> None:
        self.metadata = metadata
        self.model_config = dict(model_config)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.module = module.to(self.device)
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0

    def fit(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        y: IntArray,
        *,
        X_num_valid: FloatArray | None,
        X_cat_valid: IntArray | None,
        y_valid: IntArray | None,
        class_weights: FloatArray,
        fit_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Delegate the optimization loop to ``src.training``."""

        from src.training import fit_torch_model

        return fit_torch_model(
            self,
            X_num,
            X_cat,
            y,
            X_num_valid=X_num_valid,
            X_cat_valid=X_cat_valid,
            y_valid=y_valid,
            class_weights=class_weights,
            fit_config=fit_config,
        )

    def predict_proba(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        self.module.eval()
        outputs: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(X_num), batch_size):
                stop = start + batch_size
                logits = self.module(
                    _float_tensor(X_num[start:stop], self.device),
                    _long_tensor(X_cat[start:stop], self.device),
                )
                outputs.append(
                    torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
                )
        return np.concatenate(outputs, axis=0)

    def predict(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> IntArray:
        return self.predict_proba(X_num, X_cat, batch_size).argmax(axis=1)

    def get_training_history(self) -> list[dict[str, float]]:
        return [dict(row) for row in self.history]

    def get_embedding(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        self.module.eval()
        outputs: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(X_num), batch_size):
                stop = start + batch_size
                embedding = self.module.encode(
                    _float_tensor(X_num[start:stop], self.device),
                    _long_tensor(X_cat[start:stop], self.device),
                )
                outputs.append(embedding.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.module.parameters()))

    def save(self, path: Path | str) -> Path:
        destination = Path(path).with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_name": self.model_name,
                "state_dict": self.module.state_dict(),
                "metadata": asdict(self.metadata),
                "model_config": self.model_config,
                "seed": self.seed,
                "history": self.history,
                "best_epoch": self.best_epoch,
            },
            destination,
        )
        return destination

    def load(self, path: Path | str) -> None:
        payload = torch.load(
            Path(path),
            map_location=self.device,
            weights_only=True,
        )
        if payload["model_name"] != self.model_name:
            raise ValueError("Checkpoint architecture does not match wrapper")
        if payload["metadata"] != asdict(self.metadata):
            raise ValueError("Checkpoint metadata does not match wrapper")
        if payload["model_config"] != self.model_config:
            raise ValueError("Checkpoint configuration does not match wrapper")
        self.module.load_state_dict(payload["state_dict"], strict=True)
        self.module.to(self.device).eval()
        self.history = [dict(row) for row in payload.get("history", [])]
        self.best_epoch = int(payload.get("best_epoch", 0))


class TabNetWrapper:
    """Adapter for the native ``pytorch-tabnet`` classifier."""

    model_name = "tabnet"

    def __init__(
        self,
        metadata: ModelMetadata,
        model_config: Mapping[str, Any],
        device: str,
        seed: int,
    ) -> None:
        self.metadata = metadata
        self.model_config = dict(model_config)
        self.device = device
        self.seed = int(seed)
        cat_indices = list(
            range(metadata.n_num, metadata.n_num + metadata.n_cat)
        )
        self.model = TabNetClassifier(
            n_d=int(self.model_config.get("n_d", 16)),
            n_a=int(self.model_config.get("n_a", 16)),
            n_steps=int(self.model_config.get("n_steps", 4)),
            gamma=float(self.model_config.get("gamma", 1.3)),
            lambda_sparse=float(
                self.model_config.get("lambda_sparse", 1e-4)
            ),
            cat_idxs=cat_indices,
            cat_dims=list(metadata.categorical_cardinalities),
            cat_emb_dim=int(self.model_config.get("cat_emb_dim", 4)),
            output_dim=metadata.n_classes,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params={
                "lr": float(self.model_config.get("learning_rate", 2e-2)),
                "weight_decay": float(
                    self.model_config.get("weight_decay", 1e-5)
                ),
            },
            mask_type=str(self.model_config.get("mask_type", "entmax")),
            seed=self.seed,
            verbose=0,
            device_name=device,
        )
        self.history: list[dict[str, float]] = []
        self.best_epoch = 0

    def fit(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        y: IntArray,
        *,
        X_num_valid: FloatArray | None,
        X_cat_valid: IntArray | None,
        y_valid: IntArray | None,
        class_weights: FloatArray,
        fit_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Delegate native TabNet fitting to the training module."""

        from src.training import fit_tabnet_model

        return fit_tabnet_model(
            self,
            X_num,
            X_cat,
            y,
            X_num_valid=X_num_valid,
            X_cat_valid=X_cat_valid,
            y_valid=y_valid,
            class_weights=class_weights,
            fit_config=fit_config,
        )

    def predict_proba(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        del batch_size
        values = self.model.predict_proba(_tabnet_matrix(X_num, X_cat))
        return np.asarray(values, dtype=np.float32)

    def predict(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> IntArray:
        return self.predict_proba(X_num, X_cat, batch_size).argmax(axis=1)

    def get_training_history(self) -> list[dict[str, float]]:
        return [dict(row) for row in self.history]

    def get_embedding(
        self,
        X_num: FloatArray,
        X_cat: IntArray,
        batch_size: int,
    ) -> FloatArray:
        if not hasattr(self.model, "network"):
            raise RuntimeError("TabNet must be fitted before extracting embeddings")
        network = self.model.network
        network.eval()
        matrix = _tabnet_matrix(X_num, X_cat)
        outputs: list[NDArray[np.float32]] = []
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                values = torch.as_tensor(
                    matrix[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.model.device,
                )
                embedded = network.embedder(values)
                steps, _ = network.tabnet.encoder(embedded)
                representation = torch.stack(steps, dim=0).sum(dim=0)
                outputs.append(
                    representation.cpu().numpy().astype(np.float32)
                )
        return np.concatenate(outputs, axis=0)

    def parameter_count(self) -> int:
        if not hasattr(self.model, "network"):
            return 0
        return int(
            sum(parameter.numel() for parameter in self.model.network.parameters())
        )

    def save(self, path: Path | str) -> Path:
        destination = Path(path).with_suffix("")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with redirect_stdout(StringIO()):
            saved = self.model.save_model(str(destination))
        return Path(saved)

    def load(self, path: Path | str) -> None:
        self.model.load_model(str(Path(path)))
        expected_input = self.metadata.n_num + self.metadata.n_cat
        expected_indices = list(
            range(self.metadata.n_num, expected_input)
        )
        if self.model.input_dim != expected_input:
            raise ValueError("TabNet checkpoint input dimension is incompatible")
        if self.model.cat_idxs != expected_indices:
            raise ValueError("TabNet checkpoint categorical indices are incompatible")
        if self.model.cat_dims != list(self.metadata.categorical_cardinalities):
            raise ValueError("TabNet checkpoint cardinalities are incompatible")


class TabTransformerModule(nn.Module):
    """Contextualize categorical tokens before joining continuous values."""

    def __init__(
        self,
        metadata: ModelMetadata,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        d_token = int(config.get("d_token", 32))
        n_heads = int(config.get("n_heads", 4))
        n_layers = int(config.get("n_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        _validate_attention_dimensions(d_token, n_heads)
        self.n_cat = metadata.n_cat
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_token)
            for cardinality in metadata.categorical_cardinalities
        )
        if self.n_cat:
            layer = nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module | None = nn.TransformerEncoder(
                layer,
                num_layers=n_layers,
                norm=nn.LayerNorm(d_token),
                enable_nested_tensor=False,
            )
        else:
            self.transformer = None
        hidden = tuple(int(value) for value in config.get("mlp_hidden", (64, 32)))
        input_dim = metadata.n_num + metadata.n_cat * d_token
        self.feature_extractor, output_dim = _make_mlp_features(
            input_dim,
            hidden,
            dropout,
        )
        self.classifier = nn.Linear(output_dim, metadata.n_classes)

    def encode(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        blocks: list[Tensor] = [X_num]
        if self.n_cat:
            tokens = torch.stack(
                [
                    embedding(X_cat[:, index])
                    for index, embedding in enumerate(self.cat_embeddings)
                ],
                dim=1,
            )
            contextual = self.transformer(tokens)
            blocks.append(contextual.flatten(start_dim=1))
        joined = torch.cat(blocks, dim=1)
        return self.feature_extractor(joined)

    def forward(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        return self.classifier(self.encode(X_num, X_cat))


class FTTransformerModule(nn.Module):
    """Tokenize every feature and aggregate them with a CLS token."""

    def __init__(
        self,
        metadata: ModelMetadata,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        d_token = int(config.get("d_token", 32))
        n_heads = int(config.get("n_heads", 4))
        n_layers = int(config.get("n_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        _validate_attention_dimensions(d_token, n_heads)
        self.n_num = metadata.n_num
        self.n_cat = metadata.n_cat
        self.num_weight = nn.Parameter(torch.empty(self.n_num, d_token))
        self.num_bias = nn.Parameter(torch.empty(self.n_num, d_token))
        nn.init.normal_(self.num_weight, std=0.02)
        nn.init.zeros_(self.num_bias)
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_token)
            for cardinality in metadata.categorical_cardinalities
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_token),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, metadata.n_classes),
        )

    def encode(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        tokens: list[Tensor] = []
        if self.n_num:
            tokens.append(
                X_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)
                + self.num_bias.unsqueeze(0)
            )
        if self.n_cat:
            tokens.append(
                torch.stack(
                    [
                        embedding(X_cat[:, index])
                        for index, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
            )
        feature_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(len(X_num), -1, -1)
        encoded = self.transformer(torch.cat([cls, feature_tokens], dim=1))
        return encoded[:, 0]

    def forward(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        return self.classifier(self.encode(X_num, X_cat))


class SAINTModule(nn.Module):
    """Supervised, inductive SAINT with column attention only."""

    def __init__(
        self,
        metadata: ModelMetadata,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if bool(config.get("use_row_attention", False)):
            raise ValueError("TLM protocol forbids SAINT attention between rows")
        d_token = int(config.get("d_token", 32))
        n_heads = int(config.get("n_heads", 4))
        n_layers = int(config.get("n_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        numerical_hidden = int(config.get("numerical_embedding_hidden", 16))
        _validate_attention_dimensions(d_token, n_heads)
        self.n_num = metadata.n_num
        self.n_cat = metadata.n_cat
        self.num_embeddings = nn.ModuleList(
            nn.Sequential(
                nn.Linear(1, numerical_hidden),
                nn.ReLU(),
                nn.Linear(numerical_hidden, d_token),
            )
            for _ in range(self.n_num)
        )
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, d_token)
            for cardinality in metadata.categorical_cardinalities
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.column_embedding = nn.Parameter(
            torch.zeros(1, 1 + self.n_num + self.n_cat, d_token)
        )
        nn.init.normal_(self.column_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=4 * d_token,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.column_transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_token),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, metadata.n_classes),
        )

    def encode(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        tokens: list[Tensor] = []
        if self.n_num:
            tokens.append(
                torch.stack(
                    [
                        embedding(X_num[:, index : index + 1])
                        for index, embedding in enumerate(self.num_embeddings)
                    ],
                    dim=1,
                )
            )
        if self.n_cat:
            tokens.append(
                torch.stack(
                    [
                        embedding(X_cat[:, index])
                        for index, embedding in enumerate(self.cat_embeddings)
                    ],
                    dim=1,
                )
            )
        feature_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(len(X_num), -1, -1)
        all_tokens = torch.cat([cls, feature_tokens], dim=1)
        encoded = self.column_transformer(all_tokens + self.column_embedding)
        return encoded[:, 0]

    def forward(self, X_num: Tensor, X_cat: Tensor) -> Tensor:
        return self.classifier(self.encode(X_num, X_cat))


def create_model(
    model_name: str,
    *,
    task: str = "classification",
    data_metadata: ModelMetadata,
    model_config: Mapping[str, Any],
    device: str,
    seed: int,
) -> ClassificationModel:
    """Instantiate a supported multiclass model through one validated factory."""

    if task != "classification":
        raise ValueError("The TLM benchmark currently supports classification only")
    normalized = model_name.lower().strip()
    if normalized == "logistic_regression":
        return LogisticRegressionWrapper(
            data_metadata,
            model_config,
            device,
            seed,
        )
    if normalized == "tabnet":
        return TabNetWrapper(data_metadata, model_config, device, seed)
    modules: dict[str, type[nn.Module]] = {
        "tab_transformer": TabTransformerModule,
        "ft_transformer": FTTransformerModule,
        "saint_supervised": SAINTModule,
    }
    if normalized not in modules:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of "
            f"{sorted(['logistic_regression', 'tabnet', *modules])}"
        )
    module = modules[normalized](data_metadata, model_config)
    wrapper = TorchClassifierWrapper(
        data_metadata,
        model_config,
        device,
        seed,
        module,
    )
    wrapper.model_name = normalized
    return wrapper


def metadata_from_state(
    numerical_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
    categorical_cardinalities: tuple[int, ...],
    class_names: tuple[str, ...],
) -> ModelMetadata:
    """Build model metadata from one train-fitted preprocessing state."""

    return ModelMetadata(
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        categorical_cardinalities=categorical_cardinalities,
        class_names=class_names,
    )


def _make_mlp_features(
    input_dim: int,
    hidden: tuple[int, ...],
    dropout: float,
) -> tuple[nn.Module, int]:
    if not hidden:
        return nn.Identity(), input_dim
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden:
        layers.extend(
            [
                nn.Linear(previous, width),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        )
        previous = width
    return nn.Sequential(*layers), previous


def _validate_attention_dimensions(d_token: int, n_heads: int) -> None:
    if d_token <= 0 or n_heads <= 0:
        raise ValueError("Token width and number of heads must be positive")
    if d_token % n_heads:
        raise ValueError("Token width must be divisible by the number of heads")


def _float_tensor(values: FloatArray, device: torch.device) -> Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _long_tensor(values: IntArray, device: torch.device) -> Tensor:
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _tabnet_matrix(X_num: FloatArray, X_cat: IntArray) -> FloatArray:
    if X_cat.shape[1] == 0:
        return np.asarray(X_num, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(X_num, dtype=np.float32),
            np.asarray(X_cat, dtype=np.float32),
        ],
        axis=1,
    )
