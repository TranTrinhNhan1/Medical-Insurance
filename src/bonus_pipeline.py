import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Thêm đường dẫn project vào sys.path để import từ src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.config import DATA_PATH, RESULTS_PATH, TEST_SIZE, RANDOM_STATE
from src.data_processing import load_and_clean_data, get_train_test_data, build_preprocessing_pipeline, remove_outliers_iqr, add_engineered_features
from src.evaluation import evaluate_model, plot_actual_vs_predicted, plot_residuals, plot_learning_curve

def get_bonus_models() -> dict[str, any]:
    """Trả về dictionary chứa các mô hình bổ sung nâng cao."""
    return {
        'Log Transformed Linear Regression': LinearRegression(),
        'Log Transformed SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'LightGBM': LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    }

def get_bonus_model_params() -> dict[str, dict]:
    """Trả về param_grid cho GridSearchCV của Bonus Pipeline."""
    return {
        'Log Transformed Linear Regression': {},
        'Log Transformed SVR': {
            'regressor__model__C': [0.1, 10.0, 1000.0],
            'regressor__model__gamma': ['scale', 'auto']
        },
        'XGBoost': {
            'regressor__model__n_estimators': [50, 100, 200],
            'regressor__model__learning_rate': [0.05, 0.1]
        },
        'LightGBM': {
            'regressor__model__n_estimators': [50, 100, 200],
            'regressor__model__learning_rate': [0.05, 0.1],
            'regressor__model__num_leaves': [31, 50]
        }
    }

def main() -> None:
    """
    Hàm thực thi chính cho Bonus Pipeline.
    Áp dụng Log Transform cho biến mục tiêu và huấn luyện thêm XGBoost, LightGBM.
    """
    print("Đang tải và làm sạch dữ liệu...")
    df = load_and_clean_data(DATA_PATH)
    
    print("Đang xử lý ngoại lai...")
    df = remove_outliers_iqr(df, 'bmi')
    
    print("Đang tạo đặc trưng mới (Feature Engineering)...")
    df = add_engineered_features(df)
    
    print("Đang chia dữ liệu thành tập train và test...")
    X_train, X_test, y_train, y_test = get_train_test_data(
        df, target_col='charges', test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    models = get_bonus_models()
    param_grids = get_bonus_model_params()
    results = []

    for name, model in models.items():
        print(f"\n======================================")
        print(f"Đang huấn luyện mô hình {name}...")
        
        # Xây dựng core pipeline xử lý dữ liệu đầu vào (X)
        tree_based = name in ['XGBoost', 'LightGBM']
        preprocessor = build_preprocessing_pipeline(tree_based=tree_based)
        core_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Bọc core pipeline bằng TransformedTargetRegressor để xử lý biến mục tiêu (y)
        pipeline = TransformedTargetRegressor(
            regressor=core_pipeline,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        param_grid = param_grids.get(name, {})
        print(f"  -> Đang chạy GridSearchCV (CV=5)...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        pipeline = grid_search.best_estimator_
        cv_score = grid_search.best_score_
        if param_grid:
            print(f"  -> Best params: {grid_search.best_params_}")
        
        print(f"Đang đánh giá mô hình {name} trên Test Set...")
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        metrics['Model'] = name
        metrics['CV R2 Mean Score'] = cv_score
        results.append(metrics)
        
        print(f"Đang xuất biểu đồ cho mô hình {name}...")
        plot_actual_vs_predicted(y_test, y_pred, name)
        plot_residuals(y_test, y_pred, name)
        
        # Lưu ý: TransformedTargetRegressor không tương thích trực tiếp với hàm learning_curve của scikit-learn
        # ở dạng trực quan hóa tự động, nên ta tạm bỏ qua learning_curve cho các mô hình bọc ở bonus step này
        # để code gọn nhẹ.
        
    print("\n============== Bonus Kết Quả ==============")
    results_df = pd.DataFrame(results).set_index('Model')
    results_df = results_df[['CV R2 Mean Score', 'MAE', 'MSE', 'RMSE', 'R2 Score']]
    print(results_df)
    
    bonus_results_path = PROJECT_ROOT / 'outputs' / 'bonus_results.csv'
    print(f"\nĐang lưu kết quả vào {bonus_results_path}...")
    results_df.to_csv(bonus_results_path)
    print("Bonus Pipeline đã hoàn thành thành công!")

if __name__ == "__main__":
    main()
