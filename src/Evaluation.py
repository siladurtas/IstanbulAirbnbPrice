import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

def evaluate_models(rf_rmse, rf_r2, xgb_rmse, xgb_r2, svm_rmse, svm_r2,
                   knn_rmse, knn_r2, lasso_rmse, lasso_r2,
                   stacking_rmse, stacking_r2):
    """
    Tüm modellerin performansını değerlendirir ve en iyi modeli döndürür.
    """
    models = {
        'Random Forest': {'RMSE': rf_rmse, 'R2': rf_r2},
        'XGBoost': {'RMSE': xgb_rmse, 'R2': xgb_r2},
        'SVM': {'RMSE': svm_rmse, 'R2': svm_r2},
        'KNN': {'RMSE': knn_rmse, 'R2': knn_r2},
        'Lasso': {'RMSE': lasso_rmse, 'R2': lasso_r2},
        'Stacking': {'RMSE': stacking_rmse, 'R2': stacking_r2}
    }
    
    best_model = min(models.items(), key=lambda x: (x[1]['RMSE'], -x[1]['R2']))[0]
    
    return best_model, models

def save_results_to_csv(models: dict, path: str = 'outputs/model_results.csv'):
    df = pd.DataFrame([
        {'Model': name, 'RMSE': rmse, 'R2': r2}
        for name, (rmse, r2) in models.items()
    ])
    df.to_csv(path, index=False)
    print(f"\nModel sonuçları CSV olarak kaydedildi: {path}")

def plot_model_results(models):
    """
    Model performanslarını görselleştirir.
    """
    model_names = list(models.keys())
    rmse_values = [models[model]['RMSE'] for model in model_names]
    r2_values = [models[model]['R2'] for model in model_names]
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # RMSE grafiği
    sns.barplot(x=model_names, y=rmse_values, ax=ax1, palette='viridis')
    ax1.set_title('Model RMSE Değerleri')
    ax1.set_xlabel('Modeller')
    ax1.set_ylabel('RMSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R2 grafiği
    sns.barplot(x=model_names, y=r2_values, ax=ax2, palette='viridis')
    ax2.set_title('Model R2 Değerleri')
    ax2.set_xlabel('Modeller')
    ax2.set_ylabel('R2')
    ax2.tick_params(axis='x', rotation=45)
    

    plt.tight_layout()
    plt.savefig('outputs/model_performance.png')
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, model_name):
    """
    Gerçek vs Tahmin grafiğini çizer.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Gerçek Değerler')
    plt.ylabel('Tahmin Edilen Değerler')
    plt.title(f'{model_name} - Gerçek vs Tahmin')
    
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_actual_vs_predicted.png')
    plt.close()

def plot_residuals(y_true, y_pred, model_name):
    """
    Artık (residual) dağılımı grafiğini çizer.
    """
    residuals = y_true - y_pred
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Artık Değerler')
    plt.ylabel('Frekans')
    plt.title(f'{model_name} - Artık Dağılımı')
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Tahmin Edilen Değerler')
    plt.ylabel('Artık Değerler')
    plt.title(f'{model_name} - Artık vs Tahmin')
    
    plt.tight_layout()
    plt.savefig(f'outputs/{model_name}_residuals.png')
    plt.close()

def plot_all_model_predictions(y_true, predictions_dict):
    """
    Tüm modellerin tahminlerini karşılaştırmalı olarak görselleştirir.
    """
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items(), 1):
        plt.subplot(2, 3, i)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Gerçek Değerler')
        plt.ylabel('Tahmin Edilen Değerler')
        plt.title(f'{model_name} - Gerçek vs Tahmin')
        
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.savefig('outputs/all_models_predictions.png')
    plt.close()

def plot_all_model_residuals(y_true, predictions_dict):
    """
    Tüm modellerin artık dağılımlarını karşılaştırmalı olarak görselleştirir.
    """
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, y_pred) in enumerate(predictions_dict.items(), 1):
        residuals = y_true - y_pred
        
        plt.subplot(2, 3, i)
        sns.histplot(residuals, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Artık Değerler')
        plt.ylabel('Frekans')
        plt.title(f'{model_name} - Artık Dağılımı')
    
    plt.tight_layout()
    plt.savefig('outputs/all_models_residuals.png')
    plt.close()

    
