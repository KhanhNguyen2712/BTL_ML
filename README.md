# 🚗 German Used Car Price Prediction

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-latest-red.svg)](https://xgboost.readthedocs.io/)

Bài tập lớn Machine Learning dự đoán giá xe cũ tại thị trường Đức sử dụng các thuật toán Regression. Dự án so sánh 6 mô hình ML khác nhau và thực hiện hyperparameter tuning để đạt được hiệu suất tốt nhất.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Dataset](#-dataset)
- [Cài đặt môi trường](#️-cài-đặt-môi-trường)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
- [Quy trình thực hiện](#-quy-trình-thực-hiện)
- [Kết quả](#-kết-quả)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Lưu ý quan trọng](#-lưu-ý-quan-trọng)
- [Tác giả](#-tác-giả)

---

## 🎯 Giới thiệu

Bài tập lớn này xây dựng mô hình Machine Learning để dự đoán giá xe cũ tại thị trường Đức dựa trên các đặc điểm kỹ thuật và thông tin xe. Mục tiêu là tìm ra mô hình có độ chính xác cao nhất và hiểu được các yếu tố ảnh hưởng đến giá xe.

## 🎯 Mục tiêu chính:

- Phân tích và tiền xử lý dữ liệu xe cũ từ thị trường Đức
- So sánh hiệu suất của 6 mô hình ML: XGBoost, Random Forest, Gradient Boosting, Decision Tree, Linear Regression, KNN
- Tối ưu hóa hyperparameters cho 2 mô hình: KNN và Random Forest
- Đạt được R² score > 0.80 trên tập test

---

## 📊 Dataset

### Thông tin chung

- **Tên dataset**: German Used Car Dataset (`autos.csv`)
- **Số lượng records**: 371,528 xe
- **Số lượng features**: 20 cột
- **Kích thước**: ~56.7 MB
- **Target variable**: `price` (Giá xe, đơn vị €)
- **Nguồn**: [Kaggle - Used Cars Dataset](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices/data)
### Các features chính

| Feature               | Kiểu dữ liệu | Mô tả                           |
| --------------------- | ------------ | ------------------------------- |
| `dateCrawled`         | Date         | Ngày thu thập dữ liệu           |
| `name`                | String       | Tên xe                          |
| `seller`              | String       | Loại người bán (private/dealer) |
| `offerType`           | String       | Loại rao bán                    |
| `price`               | Integer      | **Giá xe (Target)**             |
| `abtest`              | String       | A/B testing group               |
| `vehicleType`         | String       | Loại xe (SUV, sedan, v.v.)      |
| `yearOfRegistration`  | Integer      | Năm đăng ký xe                  |
| `gearbox`             | String       | Loại hộp số (manual/automatic)  |
| `powerPS`             | Integer      | Công suất động cơ (PS)          |
| `model`               | String       | Model xe                        |
| `kilometer`           | Integer      | Số km đã đi                     |
| `monthOfRegistration` | Integer      | Tháng đăng ký                   |
| `fuelType`            | String       | Loại nhiên liệu                 |
| `brand`               | String       | Hãng xe                         |
| `notRepairedDamage`   | String       | Tình trạng hư hỏng              |
| `dateCreated`         | Date         | Ngày tạo quảng cáo              |
| `nrOfPictures`        | Integer      | Số lượng hình ảnh               |
| `postalCode`          | Integer      | Mã bưu điện                     |
| `lastSeen`            | Date         | Lần cuối thấy quảng cáo         |

### Vấn đề trong dữ liệu

- ❌ **Missing values**: 2.5% (184,008 cells)
  - `notRepairedDamage`: 19.4%
  - `vehicleType`: 10.2%
  - `fuelType`: 9.0%
- ❌ **Zero values**:
  - `price`: 10,778 (2.9%)
  - `powerPS`: 40,820 (11.0%)
- ❌ **Outliers**: Skewness cao trong `price` (gamma_1 = 578.06)
- ❌ **Duplicates**: 4 rows
- ❌ **High cardinality**: `brand` (40 unique), `model` (240+ unique)

---

## ⚙️ Cài đặt môi trường

### Yêu cầu hệ thống

- **Python**: 3.12+
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB)
- **Disk space**: Tối thiểu 500MB
- **OS**: Windows/Linux/MacOS

### Phương pháp 1: Sử dụng Conda (Khuyến nghị)

```bash
# 1. Clone repository (nếu có)
git clone https://github.com/KhanhNguyen2712/BTL_ML.git

# 2. Tạo môi trường conda từ file environments.yml
conda env create -f environments.yml

# 3. Kích hoạt môi trường
conda activate ml_btl

# 4. Kiểm tra cài đặt
python --version
jupyter --version
```

### Phương pháp 2: Sử dụng pip + venv

```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Cài đặt các thư viện
pip install -r requirements.txt

# 4. Kiểm tra cài đặt
pip list
```

### Các thư viện chính

```
numpy                 # Tính toán số học
pandas                # Xử lý dữ liệu
matplotlib            # Visualization
seaborn               # Statistical visualization
scikit-learn          # Machine Learning framework
xgboost               # Gradient Boosting
ydata-profiling       # Automated EDA
category_encoders     # Target Encoding
tqdm                  # Progress bar
jupyter               # Notebook environment
```

---

## 📁 Cấu trúc thư mục

```
BTL/
│
├── data/
│   ├── autos.csv              # Dataset chính (371,528 rows)
│   └── data_desc.md           # Mô tả chi tiết dataset
│
├── price_prediction.ipynb     # Notebook chính (code gốc)
├── requirements.txt           # Dependencies cho pip
├── environments.yml           # Environment config cho conda
│
└── README.md                  # Mô tả cho toàn bộ bài tập lớn
```

---

## 🚀 Hướng dẫn chạy

### 1. Chuẩn bị dữ liệu

```bash
# Đảm bảo file autos.csv nằm trong thư mục data/
ls data/autos.csv
```

### 2. Khởi động Jupyter Notebook

```bash
# Kích hoạt môi trường (nếu chưa)
conda activate ml_btl

# Khởi động Jupyter
jupyter notebook
```

### 3. Chạy notebook

1. Mở file `price_prediction.ipynb`
2. Chạy từng cell theo thứ tự từ trên xuống dưới


---

## 🔬 Quy trình thực hiện

### 1️⃣ **Exploratory Data Analysis (EDA)**

- Load dataset từ `data/autos.csv`
- Kiểm tra thông tin cơ bản: shape, dtypes, missing values
- Tạo báo cáo tự động với `ydata_profiling`
- Phân tích correlation, distribution, outliers

### 2️⃣ **Data Preprocessing**

#### Bước 1: Basic Cleaning

```python
# Convert datetime columns
df["dateCrawled"] = pd.to_datetime(df["dateCrawled"])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Drop useless columns
df.drop(columns=["nrOfPictures", "seller", "offerType"], inplace=True)
```

#### Bước 2: Translation (German → English)

- Dịch `gearbox`: manuell → Manual, automatik → Automatic
- Dịch `fuelType`: benzin → Petrol, diesel → Diesel
- Dịch `vehicleType`: kleinwagen → Small Car, v.v.
- Standardize `brand` names

#### Bước 3: Outlier Removal (IQR Method)

```python
# Remove outliers cho price, powerPS, yearOfRegistration
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
bounds = [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
```

#### Bước 4: Missing Values & Feature Selection

- Drop rows với missing values trong critical features
- Xóa features có correlation thấp với `price`
- Drop datetime columns (không cần cho prediction)

### 3️⃣ **Feature Engineering**

```python
# Numerical features: StandardScaler
numerical_features = ['yearOfRegistration', 'powerPS', 'kilometer']

# Low cardinality: OneHotEncoder
categorical_low = ['vehicleType', 'fuelType', 'gearbox', 'notRepairedDamage']

# High cardinality: TargetEncoder
categorical_high = ['brand', 'model']
```

### 4️⃣ **Model Training & Evaluation**

**6 mô hình được so sánh:**

1. ✅ XGBoost Regressor
2. ✅ Random Forest Regressor
3. ✅ Gradient Boosting Regressor
4. ✅ Decision Tree Regressor
5. ✅ Linear Regression
6. ✅ K-Nearest Neighbors

**Evaluation strategy:**

- **K-Fold Cross-Validation** (k=5)
- **Metrics**: MAE, RMSE, R²
- **Scoring**: Train & Test scores để phát hiện overfitting

### 5️⃣ **Hyperparameter Tuning**

#### KNN (GridSearchCV)

```python
param_grid = {
    'n_neighbors': [3, 5],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # Manhattan vs Euclidean
}
```

#### Random Forest (RandomizedSearchCV)

```python
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

### 6️⃣ **Final Testing**

- Train/Test split (80/20)
- Train mô hình với best parameters
- Đánh giá trên test set

---

## 📈 Kết quả

### Model Comparison (Cross-Validation)

| Model             | Test R²    | Test MAE (€) | Test RMSE (€) | Fit Time (s) |
| ----------------- | ---------- | ------------ | ------------- | ------------ |
| Random Forest     | 0.8772     | 903.5        | 1367.7        | 25.9         |
| XGBoost           | 0.8859     | 882.3        | 1321.4        | 0.11         |
| Gradient Boosting | 0.8473     | 1040.0       | 1528.5        | 26.5         |
| KNN               | 0.8530     | 992.3        | 1499.6        | 0.43         |
| Linear Regression | 0.7150     | 1554.9       | 2088.2        | 0.6          |
| Decision Tree     | 0.8264     | 1039.6       | 1629.5        | 1.08         |

_Lưu ý: Kết quả có thể khác nhau tùy thuộc vào preprocessing và tuning_

### Best Model Performance

**Random Forest (After Tuning):**

- ✅ R² Score: **0.8831**
- ✅ MAE: **882.3**
- ✅ RMSE: **1331.6**

**Ý nghĩa:**

- Mô hình giải thích được **88.31%** sự biến thiên của giá xe
- Sai số trung bình khoảng **882.3€** (khá tốt cho dữ liệu xe cũ)

---

## 🛠️ Công nghệ sử dụng

### Machine Learning

- **scikit-learn**: Pipeline, ColumnTransformer, Cross-Validation
- **XGBoost**: Gradient Boosting implementation
- **category_encoders**: Target Encoding cho high cardinality

### Data Processing

- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Visualization

- **matplotlib**: Static plots
- **seaborn**: Statistical visualization
- **ydata-profiling**: Automated EDA reports

### Environment

- **Jupyter Notebook**: Interactive development
- **tqdm**: Progress tracking
- **conda/pip**: Package management

---

## ⚠️ Lưu ý quan trọng

### 1. Về Data Leakage

```python
# ❌ WRONG: Fit trên toàn bộ dataset trước khi split
preprocessor.fit(X)
X_train, X_test = train_test_split(X)

# ✅ CORRECT: Fit chỉ trên training set
X_train, X_test = train_test_split(X)
preprocessor.fit(X_train)
```

### 2. Về Cross-Validation

- **Chỉ sử dụng training data** cho CV
- Không bao giờ dùng test set trong CV
- Final evaluation luôn trên holdout test set

### 3. Về Memory & Performance

```python
# Tối ưu hóa memory cho dataset lớn:
df = pd.read_csv('data/autos.csv',
                 dtype={'postalCode': 'int32', 'powerPS': 'int16'})

# Sử dụng parallel processing:
model = RandomForestRegressor(n_jobs=-1)  # Dùng tất cả CPU cores
```

### 4. Về Target Encoding

- **Chỉ dùng cho high cardinality** (brand, model)
- Có thể gây overfitting nếu lạm dụng
- Luôn kết hợp với regularization

### 5. Về Profiling Report

```python
# ydata_profiling có thể rất chậm với dataset lớn
# Nếu quá lâu, có thể giảm kích thước mẫu:
sample_df = df.sample(n=50000, random_state=42)
profile = ydata_profiling.ProfileReport(sample_df)
```

### 6. Thứ tự chạy code

⚠️ **BẮT BUỘC chạy tuần tự từ trên xuống dưới**

- Không skip các cell preprocessing
- Không chạy lại cell train mà chưa reset kernel
- Nếu gặp lỗi, restart kernel và chạy lại từ đầu

### 7. Troubleshooting

#### Lỗi: `ModuleNotFoundError: No module named 'ydata_profiling'`

```bash
pip install ydata-profiling
# hoặc
conda install -c conda-forge ydata-profiling
```

#### Lỗi: `MemoryError` khi chạy profiling

```python
# Giảm số lượng mẫu hoặc tắt một số features:
profile = ydata_profiling.ProfileReport(
    df,
    minimal=True,  # Chế độ tối giản
    explorative=False
)
```

#### Lỗi: Cross-validation quá chậm

```python
# Giảm n_splits hoặc giảm kích thước dataset:
N_SPLITS = 3  # Thay vì 5
# hoặc
X_sample, y_sample = X.sample(n=100000), y.sample(n=100000)
```

---

## 📚 Tài liệu tham khảo

- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Target Encoding](https://contrib.scikit-learn.org/category_encoders/targetencoder.html)
- [Cross-Validation Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Feature Engineering Guide](https://www.kaggle.com/learn/feature-engineering)

---

## 👨‍💻 Tác giả: Nguyễn Minh Khánh - 2311518

**Tên dự án**: German Used Car Price Prediction  
**Mục đích**: Học tập và nghiên cứu Machine Learning  
**Năm thực hiện**: 2025


