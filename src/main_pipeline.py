import pandas as pd
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from sklearn.model_selection import GridSearchCV
from src.config import DATA_PATH, RESULTS_PATH, TEST_SIZE, RANDOM_STATE
from src.data_processing import load_and_clean_data, get_train_test_data, remove_outliers_iqr, add_engineered_features
from src.model_training import get_models, build_full_pipeline, get_model_params
from src.evaluation import evaluate_model, plot_actual_vs_predicted, plot_residuals, plot_feature_importance, plot_learning_curve

def main() -> None:
    """
    Hàm thực thi chính cho pipeline học máy.
    Tải dữ liệu, huấn luyện nhiều mô hình, đánh giá, và lưu kết quả cùng với biểu đồ.
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
    
    models = get_models()
    param_grids = get_model_params()
    results = []

    for name, model in models.items():
        print(f"\n======================================")
        print(f"Đang huấn luyện mô hình {name}...")
        pipeline = build_full_pipeline(model, name)
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
        
        # Feature importance cho mô hình Random Forest và Decision Tree
        if name in ['Random Forest', 'Decision Tree']:
            # Lấy tên cột sau khi đi qua pipeline tiền xử lý
            preprocessor = pipeline.named_steps['preprocessor']
            # Lấy trực tiếp biến dạng số
            num_features = preprocessor.transformers_[0][2]
            # Lấy tên biến phân loại sau khi đi qua OneHotEncoder
            cat_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(preprocessor.transformers_[1][2])
            feature_names = list(num_features) + list(cat_features)
            plot_feature_importance(pipeline, feature_names, name)

        # Vẽ Learning Curve
        print(f"Đang xuất Learning Curve cho mô hình {name}...")
        plot_learning_curve(pipeline, X_train, y_train, name)

    print("\n====================== Kết Quả =========================")
    results_df = pd.DataFrame(results).set_index('Model')
    # Sắp xếp lại các cột theo thứ tự chuẩn
    results_df = results_df[['CV R2 Mean Score', 'MAE', 'MSE', 'RMSE', 'R2 Score']]
    print(results_df)
    
    print(f"\nĐang lưu kết quả vào {RESULTS_PATH}...")
    results_df.to_csv(RESULTS_PATH)
    print("Pipeline đã hoàn thành thành công!")

if __name__ == "__main__":
    main()
