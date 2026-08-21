from .elastic_net_model import ElasticNetAblationModel
from .gradient_boosting_model import GradientBoostingAblationModel
from .linear_regression import LinearRegressionAblationModel
from .random_forest_model import RandomForestAblationModel

MODEL_REGISTRY = {
    RandomForestAblationModel.model_type: RandomForestAblationModel,
    GradientBoostingAblationModel.model_type: GradientBoostingAblationModel,
    LinearRegressionAblationModel.model_type: LinearRegressionAblationModel,
    ElasticNetAblationModel.model_type: ElasticNetAblationModel,
}

__all__ = ["MODEL_REGISTRY"]
