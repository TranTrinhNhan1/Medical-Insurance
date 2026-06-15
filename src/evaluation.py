import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from src.config import FIGURES_DIR

def evaluate_model(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Tính toán và trả về các chỉ số đánh giá mô hình hồi quy.

    Args:
        y_true (pd.Series | np.ndarray): Giá trị thực tế của biến mục tiêu.
        y_pred (np.ndarray): Giá trị dự đoán từ mô hình.

    Returns:
        dict[str, float]: Dictionary chứa các chỉ số MAE, MSE, RMSE, và R2 Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2}

def _apply_custom_style(ax, title: str, xlabel: str, ylabel: str):
    """Áp dụng phong cách vẽ biểu đồ chuẩn của project."""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Ẩn viền trên và phải
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Thêm lưới
    ax.grid(True, alpha=0.5, linestyle=":")

def plot_actual_vs_predicted(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Vẽ biểu đồ Actual vs Predicted và lưu hình ảnh.

    Args:
        y_true (pd.Series | np.ndarray): Giá trị thực tế.
        y_pred (np.ndarray): Giá trị dự đoán.
        model_name (str): Tên mô hình.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.6, color='#2ab0ff', edgecolors='w', s=50)
    
    # Đường thẳng y=x hoàn hảo
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    
    _apply_custom_style(
        ax, 
        title=f'Actual vs Predicted - {model_name}',
        xlabel='Actual',
        ylabel='Predicted'
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'actual_vs_predicted_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_residuals(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Vẽ biểu đồ phân phối Residuals và lưu hình ảnh.

    Args:
        y_true (pd.Series | np.ndarray): Giá trị thực tế.
        y_pred (np.ndarray): Giá trị dự đoán.
        model_name (str): Tên mô hình.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(residuals, kde=True, color='#fc8d62', ax=ax, edgecolor='w')
    
    _apply_custom_style(
        ax, 
        title=f'Residuals Distribution - {model_name}',
        xlabel='Residuals',
        ylabel='Frequency'
    )
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'residuals_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()

def plot_feature_importance(pipeline: Pipeline, feature_names: list[str], model_name: str = 'Random Forest') -> None:
    """
    Vẽ biểu đồ Feature Importances cho các mô hình dạng cây.

    Args:
        pipeline (Pipeline): Pipeline đã được huấn luyện chứa mô hình.
        feature_names (list[str]): Danh sách tên các đặc trưng sau khi tiền xử lý.
        model_name (str): Tên mô hình. Mặc định là 'Random Forest'.
    """
    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=importances[indices], 
            y=np.array(feature_names)[indices], 
            hue=np.array(feature_names)[indices], 
            palette='Set2', 
            legend=False,
            ax=ax
        )
        
        _apply_custom_style(
            ax, 
            title=f'Feature Importance - {model_name}',
            xlabel='Relative Importance',
            ylabel='Features'
        )
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'feature_importance_{model_name.replace(" ", "_")}.png', dpi=300)
        plt.close()

def plot_learning_curve(pipeline: Pipeline, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, model_name: str) -> None:
    """
    Vẽ Learning Curve để kiểm tra mô hình có bị overfitting hay underfitting không.

    Args:
        pipeline (Pipeline): Pipeline chứa mô hình chưa huấn luyện hoặc đã huấn luyện.
        X (pd.DataFrame | np.ndarray): Tập dữ liệu đặc trưng (toàn bộ hoặc tập train).
        y (pd.Series | np.ndarray): Biến mục tiêu.
        model_name (str): Tên mô hình.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=5, n_jobs=-1, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    # Chuyển đổi điểm MSE bị âm thành dương (do scikit-learn dùng negative MSE để tối ưu)
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(train_sizes, train_scores_mean, 'o-', color="#8da0cb", label="Train MSE", lw=2, markersize=8)
    ax.plot(train_sizes, test_scores_mean, 'o-', color="#fc8d62", label="CV MSE", lw=2, markersize=8)
    
    _apply_custom_style(
        ax, 
        title=f'Learning Curve - {model_name}',
        xlabel='Training Examples',
        ylabel='Mean Squared Error (MSE)'
    )
    
    ax.legend(loc="best", fontsize=11, frameon=False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'learning_curve_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()
