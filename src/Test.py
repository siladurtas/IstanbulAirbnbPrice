import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_models():
    models = {}
    model_files = [
        'KNN_20250606_023415.joblib',
        'Lasso_20250606_023417.joblib',
        'RandomForest_20250606_021124.joblib',
        'Stacking_20250606_023435.joblib',
        'XGBoost_20250606_022050.joblib',
        'SVM_20250606_023405.joblib'
    ]
    
    for model_file in model_files:
        try:
            model = joblib.load(f'outputs/{model_file}')
            model_name = model_file.split('_')[0]
            models[model_name] = model
            print(f"{model_name} modeli başarıyla yükleniyor...")
        except Exception as e:
            print(f"Hata: {model_file} yüklenirken bir sorun oluştu: {str(e)}")
    
    return models

def preprocess_input(data):
    """Girdi verilerini ön işle"""
    # Seçilen özellikler
    selected_features = ['price_per_bedroom', 'accommodates', 'bedrooms', 'minimum_nights']
    
    # price sütununu baştan kaldır
    if 'price' in data.columns:
        data = data.drop('price', axis=1)
    
    # Eksik özellikleri kontrol et ve ekle
    for feature in selected_features:
        if feature not in data.columns:
            # Eksik özellik varsa ortalama değeri koy
            data[feature] = 0
    
    # Sadece seçilen özellikleri al
    data = data[selected_features]
    
    # Eksik değerleri doldur
    data = data.fillna(data.mean())
    
    # Aykırı değerleri temizle
    for col in selected_features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Kaydedilmiş StandardScaler'ı yükle
    try:
        scaler_info = joblib.load('outputs/standard_scaler.joblib')
        scaler = scaler_info['scaler']
        
        # Veriyi ölçekle
        data_scaled = pd.DataFrame(
            scaler.transform(data),
            columns=selected_features,
            index=data.index
        )
        print("Veri başarıyla ölçeklendi.")
    except Exception as e:
        print(f"Uyarı: StandardScaler yüklenemedi: {str(e)}")
        print("Ölçeklendirme yapılmadan devam ediliyor...")
        data_scaled = data
    
    return data_scaled

def calculate_derived_features(df):
    """Hesaplanmış özellikleri oluştur"""
    # Seçilen özellikler
    selected_features = ['price_per_bedroom', 'accommodates', 'bedrooms', 'minimum_nights']
    
    # Eksik özellikleri ekle
    for feature in selected_features:
        if feature not in df.columns:
            if feature == 'price_per_bedroom':
                df[feature] = 1000  # Geçici değer
            else:
                df[feature] = 0
    
    # Özellikleri doğru sırada döndür
    return df[selected_features]

def make_predictions(models, input_data):
    predictions = {}
    processed_data = preprocess_input(input_data.copy())
    
    for model_name, model in models.items():
        try:
            pred = model.predict(processed_data)
            pred_price = np.expm1(pred[0])
            predictions[model_name] = pred_price
            print(f"\n{model_name} Tahminleri:")
            print(f"Tahmin Edilen Fiyat: {pred_price:.2f} TL")
        except Exception as e:
            print(f"Hata: {model_name} modeli tahmin yaparken bir sorun oluştu: {str(e)}")
    return predictions

def get_test_data():
    # Sadece 4 özellik
    features = {
        'price_per_bedroom': 1000,  # geçici, gerçek veride güncellenecek
        'accommodates': 3,
        'bedrooms': 1,
        'minimum_nights': 3
    }
    data = pd.DataFrame([features])
    data = calculate_derived_features(data)
    return data

def main():
    print("Modeller yükleniyor...")
    models = load_models()
    
    if not models:
        print("Hiçbir model yüklenemedi. Program sonlandırılıyor.")
        return
    
    try:
        print("\nTest verisi hazırlanıyor...")
        data = get_test_data()
        
        predictions = make_predictions(models, data)
        
        results_df = pd.DataFrame(predictions, index=[0])
        results_df.to_csv('outputs/predictions.csv', index=False)
        print("\nTahminler 'outputs/predictions.csv' dosyasına kaydedildi.")
    except Exception as e:
        print(f"Hata: {str(e)}")

if __name__ == "__main__":
    main()
