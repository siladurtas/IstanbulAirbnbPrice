import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import folium
from folium.plugins import MarkerCluster

def exploration(df):

    plt.rcParams['figure.figsize'] = (18, 14)
    plt.rcParams['font.size'] = 7
    sns.set_style('whitegrid')


    if df['price'].dtype == 'object':
        df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).astype(float)

    print("\nVeri seti genel bilgisi:")
    df.info()

    # --- 1. EKSİK VERİ ANALİZİ ---
    print("\n2. Eksik Veri Analizi")
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data / len(df)) * 100
    missing_info = pd.DataFrame({
        'Missing Values': missing_data,
        'Percentage (%)': missing_percentage
    })
    print(missing_info[missing_info['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False))

    # Eksik veri yüzdeleri grafiği
    missing_percent = missing_info['Percentage (%)'][missing_info['Percentage (%)'] > 0].sort_values(ascending=False)
    missing_percent.plot(kind='bar', color='salmon')
    plt.title('Eksik Veri Yüzdeleri')
    plt.ylabel('Yüzde (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/missing_data_percentage.png')
    plt.close()

    # --- 2. KATEGORİK DEĞİŞKENLER ANALİZİ ---
    print("\n4. Kategorik Değişkenler")
    categorical_columns = ['neighbourhood', 'room_type', 'property_type']
    for col in categorical_columns:
        if col in df.columns:
            print(f"\n{col} en sık değerler:")
            print(df[col].value_counts().head(10))
            
            plt.figure(figsize=(12, 6))
            df[col].value_counts().head(10).plot(kind='bar', color='cornflowerblue')
            plt.title(f'Top 10 {col} Dağılımı')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'outputs/{col}_distribution.png')
            plt.close()

    # --- 3. KORELASYON MATRİSİ ---
    print("\n3. Correlation Analysis")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_columns].corr().fillna(0)
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('outputs/correlation_matrix.png')
    plt.close()


    # --- 4. HARİTA GÖRSELLEŞTİRMESİ ---
    print("\n7. Harita Görselleştirmesi: İstanbul'daki İlanlar")

    def price_to_color(price):
        if price < 500:
            return 'green'
        elif 500 <= price < 1000:
            return 'orange'
        else:
            return 'red'

    istanbul_price_map = folium.Map(location=[41.0082, 28.9784], zoom_start=12)
    marker_cluster = MarkerCluster().add_to(istanbul_price_map)

    for _, row in df.iterrows():
        folium.CircleMarker(
            [row['latitude'], row['longitude']],
            radius=5,
            color=price_to_color(row['price']),
            fill=True,
            fill_opacity=0.6,
            tooltip=f"Fiyat: {row['price']} ₺"
        ).add_to(marker_cluster)

    istanbul_price_map.save("outputs/istanbul_airbnb_price_map.html")

    print("\nTüm analizler ve haritalar tamamlandı! HTML ve PNG dosyalarını klasörünüzde bulabilirsiniz.")
    return df