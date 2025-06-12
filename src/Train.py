from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd
import warnings
import joblib
import os
import time
from datetime import datetime

warnings.filterwarnings("ignore")

def save_model_and_params(model, model_name, params, rmse, r2, training_time):
    """Model ve parametrelerini kaydet"""
    # Outputs klasörünü oluştur
    os.makedirs('outputs', exist_ok=True)
    
    # Modeli kaydet
    model_path = f'outputs/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
    joblib.dump(model, model_path)
    
    # Parametreleri ve performans metriklerini kaydet
    params_info = {
        'model_name': model_name,
        'parameters': params,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Parametreleri best_params.txt dosyasına ekle
    with open('outputs/best_params.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Zaman: {params_info['timestamp']}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"Eğitim Süresi: {training_time:.2f} saniye\n")
        f.write("Parametreler:\n")
        for param, value in params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"Model dosyası: {model_path}\n")
        f.write(f"{'='*50}\n")
    
    # Eğitim süresini time.txt dosyasına kaydet
    with open('outputs/time.txt', 'a', encoding='utf-8') as f:
        f.write(f"{model_name}: {training_time:.2f} saniye\n")

def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest modelini eğit"""
    print("\nRandom Forest Eğitimi Başlıyor...")
    start_time = time.time()
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    save_model_and_params(best_model, 'RandomForest', grid_search.best_params_, rmse, r2, training_time)
    
    return best_model, rmse, r2

def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost modelini eğit"""
    print("\nXGBoost Eğitimi Başlıyor...")
    start_time = time.time()
    
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    save_model_and_params(best_model, 'XGBoost', grid_search.best_params_, rmse, r2, training_time)
    
    return best_model, rmse, r2

def train_svm(X_train, y_train, X_test, y_test):
    """SVM modelini eğit"""
    print("\nSVM Eğitimi Başlıyor...")
    start_time = time.time()
    
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.3],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto', 0.1, 0.01]
    }
    
    model = SVR()
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    save_model_and_params(best_model, 'SVM', grid_search.best_params_, rmse, r2, training_time)
    
    return best_model, rmse, r2

def train_knn(X_train, y_train, X_test, y_test):
    """KNN modelini eğit"""
    print("\nKNN Eğitimi Başlıyor...")
    start_time = time.time()
    
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],  # Manhattan ve Euclidean mesafeleri
        'algorithm': ['auto', 'ball_tree', 'kd_tree']
    }
    
    model = KNeighborsRegressor()
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    save_model_and_params(best_model, 'KNN', grid_search.best_params_, rmse, r2, training_time)
    
    return best_model, rmse, r2

def train_lasso(X_train, y_train, X_test, y_test):
    """Lasso modelini eğit"""
    print("\nLasso Eğitimi Başlıyor...")
    start_time = time.time()
    
    param_grid = {
        'alpha': [0.1, 1.0, 10.0, 100.0],
        'fit_intercept': [True, False],
        'selection': ['cyclic', 'random']
    }
    
    model = Lasso(random_state=42)
    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring='neg_root_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    save_model_and_params(best_model, 'Lasso', grid_search.best_params_, rmse, r2, training_time)
    
    return best_model, rmse, r2

def train_stacking(X_train, y_train, X_test, y_test):
    """Stacking Ensemble modelini eğit"""
    print("\nStacking Ensemble Eğitimi Başlıyor...")
    start_time = time.time()
    
    # Baz modeller
    base_models = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('svm', SVR()),
        ('knn', KNeighborsRegressor()),
        ('lasso', Lasso(random_state=42))
    ]
    
    # Meta-model
    meta_model = XGBRegressor(random_state=42)
    
    # Stacking modelini oluştur
    model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3
    )
    
    # Modeli eğit
    model.fit(X_train, y_train)
    
    # Model performansını değerlendir
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    training_time = time.time() - start_time
    
    # Model ve parametreleri kaydet
    params = {
        'base_models': [name for name, _ in base_models],
        'meta_model': 'XGBoost'
    }
    save_model_and_params(model, 'Stacking', params, rmse, r2, training_time)
    
    return model, rmse, r2



