import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

def feature_selection(X: pd.DataFrame, y: pd.Series, 
                      correlation_threshold: float = 0.05, 
                      importance_threshold: float = 0.01) -> pd.DataFrame:
    """
    Korelasyon ve model tabanlı özellik seçimi uygular.

    """
    os.makedirs('outputs', exist_ok=True)

    # 1. Korelasyon tabanlı filtreleme
    print("\n1. Korelasyon tabanlı filtreleme uygulanıyor...")

    correlations = X.corrwith(y).abs()
    correlation_df = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    }).sort_values('correlation', ascending=False)

    correlation_df.to_csv('outputs/feature_correlations.csv', index=False)

    selected_features = correlations[correlations > correlation_threshold].index.tolist()
    print(f"Korelasyon filtresinden geçen özellik sayısı: {len(selected_features)}")

    if not selected_features:
        print("Uyarı: Hiçbir özellik korelasyon eşiğini geçemedi. Tüm özellikler kullanılacak.")
        selected_features = X.columns.tolist()

    X_filtered = X[selected_features]

    # 2. Model tabanlı seçim (Random Forest ile)
    print("\n2. Model tabanlı özellik seçimi uygulanıyor...")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_filtered, y)

    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    feature_importance.to_csv('outputs/feature_importance.csv', index=False)

    final_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
    print(f"Önem filtresinden geçen özellik sayısı: {len(final_features)}")

    # 3. Görselleştirme
    print("\n3. Özellik önemi ve korelasyon görselleştiriliyor...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    sns.barplot(data=feature_importance.head(10), x='importance', y='feature', ax=ax1, palette='viridis')
    ax1.set_title('En Önemli 10 Özellik (Model Tabanlı)')
    ax1.set_xlabel('Özellik Önemi')
    ax1.set_ylabel('Özellik')

    correlation_plot = correlation_df[correlation_df['feature'].isin(final_features)]
    sns.barplot(data=correlation_plot.head(10), x='correlation', y='feature', ax=ax2, palette='magma')
    ax2.set_title('Hedef Değişkenle En Yüksek Korelasyonlu 10 Özellik')
    ax2.set_xlabel('Korelasyon')
    ax2.set_ylabel('Özellik')

    plt.tight_layout()
    plt.savefig('outputs/feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Seçilen özellikleri kaydet
    with open('outputs/selected_features.txt', 'w') as f:
        f.write("Seçilen Özellikler:\n")
        f.write("=" * 50 + "\n")
        f.write(f"Toplam Seçilen Özellik Sayısı: {len(final_features)}\n")
        f.write("=" * 50 + "\n")
        for feature in final_features:
            f.write(f"{feature}\n")

    print("\nSeçilen özellikler outputs klasörüne kaydedildi:")
    print("- feature_correlations.csv")
    print("- feature_importance.csv")
    print("- feature_analysis.png")
    print("- selected_features.txt")

    return X[final_features]
