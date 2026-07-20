# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn

from configs import config


class NumericalProjection(nn.Module):

    def __init__(self,input_size: int,output_size: int,) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
        )

    def forward(self,numerical_features: torch.Tensor,) -> torch.Tensor:
        return self.projection(numerical_features)

class CategoricalEmbedding(nn.Module):

    def __init__(self,categorical_cardinalities: dict[str, int],) -> None:
        super().__init__()

        self.feature_names = list(categorical_cardinalities.keys())

        self.embeddings = nn.ModuleDict()
        self.output_size = 0

        for feature_name, cardinality in categorical_cardinalities.items():
            embedding_dim = self._calculate_embedding_dim(cardinality)

            self.embeddings[feature_name] = nn.Embedding(
                num_embeddings=cardinality + 1,
                embedding_dim=embedding_dim,
            )

            self.output_size += embedding_dim

    def _calculate_embedding_dim(self,cardinality: int,) -> int:
        return min(
            config.MAX_EMBEDDING_DIM,
            round(1.6 * (cardinality ** 0.56)),
        )

    def forward(self,categorical_features: torch.Tensor,) -> torch.Tensor:
        embedded_features = []

        categorical_features = categorical_features + 1
        categorical_features = torch.clamp(categorical_features, min=0)

        for index, feature_name in enumerate(self.feature_names):
            feature_tensor = categorical_features[:, index]

            embedded_feature = self.embeddings[feature_name](feature_tensor)

            embedded_features.append(embedded_feature)

        return torch.cat(embedded_features, dim=1)

class SharedBackbone(nn.Module):

    def __init__(self,input_size: int,) -> None:
        super().__init__()

        layers = []
        input_features = input_size

        for hidden_size in config.HIDDEN_LAYERS:
            layers.extend(
                [
                    nn.Linear(input_features, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.DROPOUT),
                ]
            )

            input_features = hidden_size

        self.backbone = nn.Sequential(*layers)
        self.output_size = config.HIDDEN_LAYERS[-1]

    def forward(self,features: torch.Tensor,) -> torch.Tensor:
        return self.backbone(features)

class PredictionHead(nn.Module):
    def __init__(self,input_size: int,output_size: int,) -> None:
        super().__init__()

        hidden_size = input_size // 2

        self.head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self,features: torch.Tensor,) -> torch.Tensor:
        return self.head(features)

class RideETANetwork(nn.Module):
    def __init__(self,numerical_feature_count: int,categorical_cardinalities: dict[str, int],) -> None:
        super().__init__()

        self.numerical_projection = NumericalProjection(
            input_size=numerical_feature_count,
            output_size=config.NUMERICAL_PROJECTION_SIZE,
        )

        self.categorical_embedding = CategoricalEmbedding(
            categorical_cardinalities=categorical_cardinalities,
        )

        backbone_input_size = (
            config.NUMERICAL_PROJECTION_SIZE
            + self.categorical_embedding.output_size
        )

        self.shared_backbone = SharedBackbone(
            input_size=backbone_input_size,
        )

        self.eta_head = PredictionHead(
            input_size=self.shared_backbone.output_size,
            output_size=config.ETA_OUTPUT_SIZE,
            )

        self.delay_head = PredictionHead(
            input_size=self.shared_backbone.output_size,
            output_size=config.DELAY_OUTPUT_SIZE,
            )

    def forward(self,numerical_features: torch.Tensor,categorical_features: torch.Tensor,) -> dict[str, torch.Tensor]:

        numerical_features = self.numerical_projection(
            numerical_features,
        )

        categorical_features = self.categorical_embedding(
            categorical_features,
        )

        shared_features = torch.cat(
            [
                numerical_features,
                categorical_features,
            ],
            dim=1,
        )

        shared_features = self.shared_backbone(
            shared_features,
        )

        return {
            "eta": nn.functional.softplus(self.eta_head(shared_features)),
            "delay": self.delay_head(shared_features),
        }