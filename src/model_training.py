from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from src.data_processing import build_preprocessing_pipeline
from src.config import RANDOM_STATE

def get_models() -> dict[str, Any]:
    """
    Trả về một dictionary chứa các mô hình cần đánh giá.

    Returns:
        dict[str, Any]: Một dictionary với key là tên mô hình và value là 
        đối tượng mô hình hồi quy của scikit-learn đã được khởi tạo.
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(random_state=RANDOM_STATE),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Support Vector Regressor': SVR()
    }
    return models

def get_model_params() -> dict[str, dict]:
    """
    Trả về một dictionary chứa param_grid cho GridSearchCV.
    Chú ý tiền tố 'model__' do chúng ta sử dụng Pipeline có tên step là 'model'.
    """
    return {
        'Linear Regression': {},
        'Decision Tree': {
            'model__max_depth': [3, 5, 10, None],
            'model__min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [5, 10, None]
        },
        'K-Nearest Neighbors': {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance']
        },
        'Support Vector Regressor': {
            'model__C': [0.1, 1.0, 10.0, 100.0, 1000.0],
            'model__gamma': ['scale', 'auto', 0.1, 0.01]
        }
    }

def build_full_pipeline(model: Any, model_name: str) -> Pipeline:
    """
    Kết hợp bước tiền xử lý và mô hình thành một Pipeline duy nhất.

    Args:
        model (Any): Đối tượng mô hình hồi quy của scikit-learn đã khởi tạo.
        model_name (str): Tên mô hình để tinh chỉnh bước tiền xử lý.

    Returns:
        Pipeline: Một Pipeline của scikit-learn bao gồm tiền xử lý và huấn luyện mô hình.
    """
    tree_based = model_name in ['Decision Tree', 'Random Forest']
    preprocessor = build_preprocessing_pipeline(tree_based=tree_based)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
