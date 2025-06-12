import pandas as pd
import numpy as np
import warnings
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Veri ön işleme fonksiyonu.
    Yeni özellikler oluşturur, aykırı değerleri temizler ve veriyi ölçekler.
    """

    # 1. Gereksiz sütunları sil
    drop_columns = [
        'id', 'listing_url', 'scrape_id', 'host_id', 'host_url', 'picture_url', 
        'host_thumbnail_url', 'host_picture_url', 'last_scraped', 'source',
        'calendar_last_scraped', 'calendar_updated', 'license',
        'name', 'description', 'neighborhood_overview', 'host_about',
        'minimum_minimum_nights', 'maximum_minimum_nights', 
        'minimum_maximum_nights', 'maximum_maximum_nights',
        'minimum_nights_avg_ntm', 'maximum_nights_avg_ntm',
        'host_listings_count', 'host_total_listings_count',
        'calculated_host_listings_count_entire_homes',
        'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms',
        'host_has_profile_pic', 'host_identity_verified',
        'host_response_rate', 'host_acceptance_rate',
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]
    df = data.drop(columns=[col for col in drop_columns if col in data.columns])

    # 2. Price sütununu temizle
    if df['price'].dtype == 'object':
        df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
    df['price'] = df['price'].fillna(df['price'].median())
    Q1, Q3 = df['price'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df['price'] = df['price'].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    df['price'] = np.log1p(df['price'])

    # 3. Object sütunları sil, room_type hariç
    room_type = df['room_type'] if 'room_type' in df.columns else None
    df = df.select_dtypes(include=['int64', 'float64'])

    # 4. %40'tan fazla eksik verisi olan sütunları sil
    missing = df.isnull().mean() * 100
    df = df.drop(columns=missing[missing >= 40].index)

    # 5. Yeni özellikler
    df['bed_per_person'] = df['beds'] / df['accommodates']
    df['price_per_bedroom'] = df['price'] / df['bedrooms']
    if 'latitude' in df.columns and 'longitude' in df.columns:
        center_lat, center_lon = df['latitude'].mean(), df['longitude'].mean()
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return 2 * R * np.arcsin(np.sqrt(a))
        df['distance_to_center'] = haversine(df['latitude'], df['longitude'], center_lat, center_lon)
    df = df.fillna(0)

    # 6. Aykırı değerleri temizle (price hariç)
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col == 'price': continue
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    """Bu dönüşüm, çok yüksek yorum sayılarının etkisini azaltmak ve dağılımı 
    daha normal hale getirmek için yapılır."""

    if 'number_of_reviews' in df.columns:
        df['number_of_reviews'] = np.log1p(df['number_of_reviews'])

    # 7. Eksik verileri doldur
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)

    # 8. room_type encoding
    if room_type is not None:
        room_type = room_type.fillna(room_type.mode()[0])
        ohe = OneHotEncoder(sparse_output=False, drop='first')
        encoded = ohe.fit_transform(room_type.values.reshape(-1, 1))
        room_df = pd.DataFrame(encoded, columns=[f'room_type_{cat}' for cat in ohe.categories_[0][1:]], index=df.index)
        df_imputed = pd.concat([df_imputed, room_df], axis=1)

    # 9. Ölçekleme
    X = df_imputed.drop('price', axis=1)
    y = df_imputed['price']
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    # 10. Korelasyon kontrolü
    correlations = X_scaled.corrwith(y).abs().sort_values(ascending=False)
    print("\nPrice ile en yüksek korelasyona sahip özellikler:")
    print(correlations.head(20))

    print("\nÖn işleme tamamlandı!")
    print(f"Final veri seti boyutu: {X_scaled.shape}")

    return X_scaled, y
