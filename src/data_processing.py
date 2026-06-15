import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_clean_data(file_path: str | Path) -> pd.DataFrame:
    """
    Đọc dữ liệu từ file CSV, làm sạch tên cột và xóa các dòng trùng lặp.

    Args:
        file_path (str | Path): Đường dẫn đến file CSV.

    Returns:
        pd.DataFrame: DataFrame đã được làm sạch.
    """
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df.drop_duplicates(inplace=True)
    return df

def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Loại bỏ các giá trị ngoại lai dựa trên phương pháp IQR.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        column (str): Tên cột cần xử lý ngoại lai.

    Returns:
        pd.DataFrame: DataFrame sau khi loại bỏ ngoại lai.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Chỉ giữ lại các dòng nằm trong khoảng [lower_bound, upper_bound]
    cleaned_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()
    print(f"Đã loại bỏ {len(df) - len(cleaned_df)} điểm ngoại lai ở cột {column}.")
    return cleaned_df

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng mới (Feature Engineering).

    Args:
        df (pd.DataFrame): DataFrame đầu vào.

    Returns:
        pd.DataFrame: DataFrame đã được thêm đặc trưng mới.
    """
    df = df.copy()
    # Tạo đặc trưng is_smoker_obese: 1 nếu hút thuốc và béo phì (bmi > 30), ngược lại 0
    df['is_smoker_obese'] = ((df['smoker'] == 'yes') & (df['bmi'] > 30)).astype(int)
    return df

def get_train_test_data(
    df: pd.DataFrame, 
    target_col: str = 'charges', 
    test_size: float = 0.2, 
    random_state: int = 9999
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Tách dữ liệu thành biến đặc trưng (X) và biến mục tiêu (y), sau đó chia thành tập train và test.

    Args:
        df (pd.DataFrame): DataFrame đầu vào.
        target_col (str): Tên cột mục tiêu. Mặc định là 'charges'.
        test_size (float): Tỷ lệ dữ liệu dành cho tập test. Mặc định là 0.2.
        random_state (int): Seed ngẫu nhiên để tái lặp kết quả. Mặc định là 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(tree_based: bool = False) -> ColumnTransformer:
    """
    Xây dựng một pipeline tiền xử lý bằng scikit-learn cho các biến số học và phân loại.

    Args:
        tree_based (bool): Nếu True, OneHotEncoder sẽ không drop cột nào (drop=None) 
                           để giữ tính đối xứng cho mô hình dạng cây. Nếu False, 
                           sử dụng drop='first' để tránh đa cộng tuyến cho mô hình tuyến tính.

    Returns:
        ColumnTransformer: Pipeline tiền xử lý bao gồm việc chuẩn hóa các biến số 
        và one-hot encode các biến phân loại.
    """
    numeric_features = ['age', 'bmi', 'children', 'is_smoker_obese']
    categorical_features = ['sex', 'smoker', 'region']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    drop_param = None if tree_based else 'first'
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop=drop_param))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    return preprocessor
