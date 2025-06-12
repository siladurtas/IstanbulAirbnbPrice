from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Model ve scaler yükleme fonksiyonları (verdiğin koddan)
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
    selected_features = ['price_per_bedroom', 'accommodates', 'bedrooms', 'minimum_nights']
    
    if 'price' in data.columns:
        data = data.drop('price', axis=1)
    
    for feature in selected_features:
        if feature not in data.columns:
            data[feature] = 0
    
    data = data[selected_features]
    data = data.fillna(data.mean())
    
    for col in selected_features:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    try:
        scaler_info = joblib.load('outputs/standard_scaler.joblib')
        scaler = scaler_info['scaler']
        data_scaled = pd.DataFrame(
            scaler.transform(data),
            columns=selected_features,
            index=data.index
        )
    except Exception as e:
        print(f"Uyarı: StandardScaler yüklenemedi: {str(e)}")
        data_scaled = data
    
    return data_scaled

def calculate_derived_features(df):
    selected_features = ['price_per_bedroom', 'accommodates', 'bedrooms', 'minimum_nights']
    for feature in selected_features:
        if feature not in df.columns:
            if feature == 'price_per_bedroom':
                df[feature] = 1000  # Geçici değer
            else:
                df[feature] = 0
    return df[selected_features]

# API başlangıcında modelleri yükle
models = load_models()

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if not models:
        return jsonify({"error": "Modeller yüklenemedi."}), 500

    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "Geçerli JSON formatında veri gönderin."}), 400
        
        input_df = pd.DataFrame([json_data])
        input_df = calculate_derived_features(input_df)
        processed_data = preprocess_input(input_df)
        
        results = {}
        for model_name, model in models.items():
            pred = model.predict(processed_data)
            pred_price = float(np.expm1(pred[0]))  # Burada float dönüştürme
            results[model_name] = round(pred_price, 2)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Tahmin sırasında hata oluştu: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

"""{ "price_per_bedroom": 1200, 
"accommodates": 3, 
"bedrooms": 2, 
"minimum_nights": 2 }"""