import pandas as pd
import numpy as np
from DataLoader import load_data
from sklearn.model_selection import train_test_split
from Preprocessing import preprocess_data
from FeatureSelection import feature_selection
from Train import (
    train_random_forest,
    train_xgboost,
    train_svm,
    train_knn,
    train_lasso,
    train_stacking
)
from Evaluation import (
    evaluate_models,
    save_results_to_csv,
    plot_model_results,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_all_model_predictions,
    plot_all_model_residuals
)
import warnings
import joblib
from sklearn.preprocessing import StandardScaler

def main():
    # Veriyi yükle
    print("Veri yükleniyor...")
    df = load_data('data/listings.csv.gz')
    print(f"Veri seti boyutu: {df.shape}")
    
    # İlk 5 satırı göster ve kaydet
    print("\nİlk 5 satır:")
    print(df.head())
    
    # Veriyi ön işle
    print("\nVeri ön işleme başlıyor...")
    X, y = preprocess_data(df)
    
    print(X.head())
    print(X.columns)

    # Özellik seçimi yap
    print("\nÖzellik seçimi başlıyor...")
    X_selected = feature_selection(X, y, correlation_threshold=0.05, importance_threshold=0.01)
    print(f"Seçilen özellik sayısı: {X_selected.shape[1]}")

    # Seçilen özellikler için yeni bir scaler oluştur ve kaydet
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_selected),
        columns=X_selected.columns,
        index=X_selected.index
    )
    
    # Scaler'ı kaydet
    scaler_info = {
        'scaler': scaler,
        'feature_names': X_selected.columns.tolist()
    }
    joblib.dump(scaler_info, 'outputs/standard_scaler.joblib')

    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"\nEğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    
    # Modelleri eğit ve değerlendir
    print("\nModel eğitimi ve değerlendirmesi başlıyor...")
    
    # Random Forest
    rf_model, rf_rmse, rf_r2 = train_random_forest(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, rf_model.predict(X_test), 'RandomForest')
    plot_residuals(y_test, rf_model.predict(X_test), 'RandomForest')
    
    # XGBoost
    xgb_model, xgb_rmse, xgb_r2 = train_xgboost(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, xgb_model.predict(X_test), 'XGBoost')
    plot_residuals(y_test, xgb_model.predict(X_test), 'XGBoost')
    
    # SVM
    svm_model, svm_rmse, svm_r2 = train_svm(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, svm_model.predict(X_test), 'SVM')
    plot_residuals(y_test, svm_model.predict(X_test), 'SVM')
    
    # KNN
    knn_model, knn_rmse, knn_r2 = train_knn(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, knn_model.predict(X_test), 'KNN')
    plot_residuals(y_test, knn_model.predict(X_test), 'KNN')
    
    # Lasso
    lasso_model, lasso_rmse, lasso_r2 = train_lasso(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, lasso_model.predict(X_test), 'Lasso')
    plot_residuals(y_test, lasso_model.predict(X_test), 'Lasso')
    
    # Stacking Ensemble
    stacking_model, stacking_rmse, stacking_r2 = train_stacking(X_train, y_train, X_test, y_test)
    plot_actual_vs_predicted(y_test, stacking_model.predict(X_test), 'Stacking')
    plot_residuals(y_test, stacking_model.predict(X_test), 'Stacking')
    
    # Model performanslarını değerlendir
    best_model, models = evaluate_models(
        rf_rmse, rf_r2,
        xgb_rmse, xgb_r2,
        svm_rmse, svm_r2,
        knn_rmse, knn_r2,
        lasso_rmse, lasso_r2,
        stacking_rmse, stacking_r2
    )
    
    # En iyi modeli göster
    print(f"\nEn iyi model: {best_model}")
    
    # Tüm modellerin tahminlerini karşılaştırmalı olarak görselleştir
    predictions = {
        'Random Forest': rf_model.predict(X_test),
        'XGBoost': xgb_model.predict(X_test),
        'SVM': svm_model.predict(X_test),
        'KNN': knn_model.predict(X_test),
        'Lasso': lasso_model.predict(X_test),
        'Stacking': stacking_model.predict(X_test)
    }
    
    # Karşılaştırmalı grafikleri çiz
    plot_all_model_predictions(y_test, predictions)
    plot_all_model_residuals(y_test, predictions)
    
    # Sonuçları kaydet ve görselleştir
    plot_model_results(models)

if __name__ == "__main__":
    main()
