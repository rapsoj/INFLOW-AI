from .elastic_net_model import ElasticNetAblationModel
from .gradient_boosting_model import GradientBoostingAblationModel
<<<<<<< HEAD
from .linear_regression import LinearRegressionAblationModel
=======
>>>>>>> origin/main
from .random_forest_model import RandomForestAblationModel

MODEL_REGISTRY = {
    RandomForestAblationModel.model_type: RandomForestAblationModel,
    GradientBoostingAblationModel.model_type: GradientBoostingAblationModel,
<<<<<<< HEAD
    LinearRegressionAblationModel.model_type: LinearRegressionAblationModel,
=======
>>>>>>> origin/main
    ElasticNetAblationModel.model_type: ElasticNetAblationModel,
}

__all__ = ["MODEL_REGISTRY"]
