import pandas as pd
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# 1. Wczytanie i przygotowanie danych
file_path = 'C:/Users/PC/Desktop/bmw/dane_finalne.xlsx'  
df = pd.read_excel(file_path, dtype=str)

print(f"Początkowa liczba wierszy w pliku: {len(df)}")
df.columns = df.columns.str.strip()

def clean_hard(x):
    if pd.isna(x) or x == 'nan': return None
    return ''.join([c for c in str(x) if c.isdigit()])

cols_to_clean = ['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col].apply(clean_hard), errors='coerce')

df = df.dropna(subset=['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg'])
df = df[(df['Cena'] >= 90000) & (df['Cena'] <= 500000)]
df = df[(df['Moc'] > 50)]

mask_nowe = df['Stan'].astype(str).str.strip().str.lower().isin(['nowe', 'nowy'])
df = df[(df['Przebieg'] > 1000) | ((df['Przebieg'] <= 1000) & mask_nowe)]

# --- INŻYNIERIA CECH ---
AKTUALNY_ROK = datetime.now().year
df['Wiek'] = AKTUALNY_ROK - df['Rocznik']
df['Przebieg_Roczny'] = df['Przebieg'] / (df['Wiek'] + 0.5)
df['Moc_Z_Pojemnosci'] = df['Moc'] / (df['Pojemność'] + 1)

df['Seria_Main'] = df['Seria/Model'].apply(lambda x: str(x).split(' ')[0] if pd.notna(x) else 'Brak')
df['Is_M_Power'] = df['Seria/Model'].apply(lambda x: 1 if 'M' in str(x).split(' ')[0] else 0)

# Odcięcie skrajnych anomalii cenowych (2% - 98%)
q_low = df['Cena'].quantile(0.02)
q_high = df['Cena'].quantile(0.98)
df = df[(df['Cena'] >= q_low) & (df['Cena'] <= q_high)]

cols_cat = ['Typ sprzedawcy', 'Kraj pochodzenia', 'Typ skrzyni', 'Typ nadwozia', 'Typ paliwa', 'Generacja', 'Seria/Model', 'Stan', 'Seria_Main']
for col in cols_cat:
    df[col] = df[col].fillna('Brak danych').astype(str)

identyfikatory_auta = ['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg', 'Seria/Model']
df = df.drop_duplicates(subset=identyfikatory_auta)

# Flagowanie anomalii
print("[INFO] Uruchamiam Isolation Forest w celu oflagowania wewnętrznych outlierów...")
features_for_anomaly = ['Wiek', 'Moc', 'Pojemność', 'Przebieg', 'Przebieg_Roczny', 'Moc_Z_Pojemnosci']
iso_forest = IsolationForest(contamination=0.03, random_state=42)
anomaly_preds = iso_forest.fit_predict(df[features_for_anomaly])
df['Is_Anomaly'] = np.where(anomaly_preds == -1, 1, 0)

features = ['Typ sprzedawcy', 'Kraj pochodzenia', 'Typ skrzyni', 'Typ nadwozia', 'Typ paliwa', 'Generacja', 'Seria/Model', 'Stan', 'Seria_Main', 'Wiek', 'Moc', 'Pojemność', 'Przebieg', 'Is_M_Power', 'Przebieg_Roczny', 'Moc_Z_Pojemnosci', 'Is_Anomaly']
cat_features = cols_cat

X = df[features]
y_log = np.log(df['Cena']) 

# Podział na zbiór treningowy i ostateczny testowy
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)


# TRENING MODELU 

print("\nTrenuję model CatBoost...")
best_params = {
    'iterations': 1200,
    'depth': 6,
    'learning_rate': 0.04,
    'l2_leaf_reg': 3.5,
    'loss_function': 'RMSE',
    'logging_level': 'Silent',
    'random_seed': 42
}
model = CatBoostRegressor(**best_params, cat_features=cat_features)
model.fit(X_train, y_train_log)


# EWALUACJA MODELU NA ZBIORZE TESTOWYM (R2, MAPE, BŁĘDY W PROCENTACH)

y_test_real = np.exp(y_test_log)
y_pred_real = np.exp(model.predict(X_test))

r2 = r2_score(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)

print("\n" + "="*50)
print("--- METRYKI EWALUACJI MODELU (ZBIÓR TESTOWY) ---")
print(f"R² Score: {r2:.4f}")
print(f"MAPE:     {mape * 100:.2f}%")
print(f"MAE:      {mae:,.2f} PLN")
print("="*50)

# Rozkład dokładności predykcji (błąd względny)
abs_percentage_errors = np.abs((y_test_real - y_pred_real) / y_test_real) * 100

within_5 = np.mean(abs_percentage_errors <= 5) * 100
within_10 = np.mean(abs_percentage_errors <= 10) * 100
within_15 = np.mean(abs_percentage_errors <= 15) * 100
within_20 = np.mean(abs_percentage_errors <= 20) * 100
within_30 = np.mean(abs_percentage_errors <= 30) * 100
within_50 = np.mean(abs_percentage_errors <= 50) * 100

print("\n--- STRUKTURA BŁĘDÓW PREDYKCJI MODELU ---")
print(f"Predykcje z błędem <=  5%: {within_5:.2f}% aut")
print(f"Predykcje z błędem <= 10%: {within_10:.2f}% aut")
print(f"Predykcje z błędem <= 15%: {within_15:.2f}% aut")
print(f"Predykcje z błędem <= 20%: {within_20:.2f}% aut")
print(f"Predykcje z błędem <= 30%: {within_30:.2f}% aut")
print(f"Predykcje z błędem <= 50%: {within_50:.2f}% aut")
print("="*50 + "\n")

# ==============================================================================
# WIZUALIZACJA: FEATURE IMPORTANCE & SHAP
# ==============================================================================
# 1. Feature Importance (CatBoost)
feature_importances = model.get_feature_importance()
fi_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title('CatBoost Feature Importance')
plt.xlabel('Waga zmiennej (%)')
plt.ylabel('Cecha')
plt.tight_layout()
plt.savefig('bmw_feature_importance.png', bbox_inches='tight', dpi=300)
plt.show()

# 2. SHAP Values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot', fontsize=14)
plt.tight_layout()
plt.savefig('bmw_shap_summary.png', bbox_inches='tight', dpi=300)
plt.show()

# ==============================================================================
# REKORD 1: BMW SERII 5 530i LUXURY LINE (BIAŁA - 2022)
# ==============================================================================
cena_1 = 147000
rocznik_1 = 2022
wiek_1 = AKTUALNY_ROK - rocznik_1
przebieg_1 = 92000
moc_1 = 252
pojemnosc_1 = 1998

przebieg_roczny_1 = przebieg_1 / (wiek_1 + 0.5)
moc_z_pojemnosci_1 = moc_1 / (pojemnosc_1 + 1)

input_1_pre = pd.DataFrame({'Wiek': [wiek_1], 'Moc': [moc_1], 'Pojemność': [pojemnosc_1], 'Przebieg': [przebieg_1], 'Przebieg_Roczny': [przebieg_roczny_1], 'Moc_Z_Pojemnosci': [moc_z_pojemnosci_1]})
is_anomaly_1 = 1 if iso_forest.predict(input_1_pre[features_for_anomaly])[0] == -1 else 0

input_data_1 = pd.DataFrame({
    'Typ sprzedawcy': ['Osoba prywatna'], 'Kraj pochodzenia': ['Polska'], 'Typ skrzyni': ['Automatyczna'], 'Typ nadwozia': ['Sedan'], 'Typ paliwa': ['Benzyna'],
    'Generacja': ['G30/G31 (2016-2023)'], 'Seria/Model': ['Seria 5'], 'Stan': ['Używane'], 'Seria_Main': ['Seria'],
    'Wiek': [wiek_1], 'Moc': [moc_1], 'Pojemność': [pojemnosc_1], 'Przebieg': [przebieg_1], 'Is_M_Power': [0],
    'Przebieg_Roczny': [przebieg_roczny_1], 'Moc_Z_Pojemnosci': [moc_z_pojemnosci_1], 'Is_Anomaly': [is_anomaly_1]
})[features]

wycena_1_pln = np.exp(model.predict(input_data_1)[0])

print(f"\n==============================================================================")
print(f"--- REKORD 1: BMW Seria 5 530i Luxury Line (Rocznik 2022) ---")
print(f"==============================================================================")
print(f"-> Cena z ogłoszenia:  {cena_1:,.2f} PLN")
print(f"-> Wycena modelu:      {wycena_1_pln:,.2f} PLN")
roznica_1 = ((wycena_1_pln - cena_1) / cena_1) * 100
print(f"-> Różnica:            {roznica_1:.2f}% (Model uważa za: {'OKAZJĘ' if roznica_1 > 0 else 'ZAWYŻONE'})")
print(f"-> Status anomalii IF: {'Wykryto odchylenie (Outlier)' if is_anomaly_1 == 1 else 'Normalny profil rynkowy'}")

bylo_w_train_1 = X_train[(X_train['Wiek'] == wiek_1) & (X_train['Przebieg'] == przebieg_1) & (X_train['Moc'] == moc_1)]
if len(bylo_w_train_1) > 0:
    print(f"-> [WERYFIKACJA]: Wykryto to auto w zbiorze TRENINGOWYM ({len(bylo_w_train_1)} szt.).")
else:
    print(f"-> [WERYFIKACJA]: Brak w zbiorze treningowym. Predykcja jest w 100% czysta!")


# ==============================================================================
# REKORD 2: BMW SERII 5 530i M SPORT (GRAFITOWA ZE KIEROWNICĄ - 2019)
# ==============================================================================
cena_2 = 103000
rocznik_2 = 2019
wiek_2 = AKTUALNY_ROK - rocznik_2
przebieg_2 = 197500
moc_2 = 252
pojemnosc_2 = 1998

przebieg_roczny_2 = przebieg_2 / (wiek_2 + 0.5)
moc_z_pojemnosci_2 = moc_2 / (pojemnosc_2 + 1)

input_2_pre = pd.DataFrame({'Wiek': [wiek_2], 'Moc': [moc_2], 'Pojemność': [pojemnosc_2], 'Przebieg': [przebieg_2], 'Przebieg_Roczny': [przebieg_roczny_2], 'Moc_Z_Pojemnosci': [moc_z_pojemnosci_2]})
is_anomaly_2 = 1 if iso_forest.predict(input_2_pre[features_for_anomaly])[0] == -1 else 0

input_data_2 = pd.DataFrame({
    'Typ sprzedawcy': ['Osoba prywatna'], 'Kraj pochodzenia': ['Polska'], 'Typ skrzyni': ['Automatyczna'], 'Typ nadwozia': ['Sedan'], 'Typ paliwa': ['Benzyna'],
    'Generacja': ['G30/G31 (2016-2023)'], 'Seria/Model': ['Seria 5'], 'Stan': ['Używane'], 'Seria_Main': ['Seria'],
    'Wiek': [wiek_2], 'Moc': [moc_2], 'Pojemność': [pojemnosc_2], 'Przebieg': [przebieg_2], 'Is_M_Power': [0],
    'Przebieg_Roczny': [przebieg_roczny_2], 'Moc_Z_Pojemnosci': [moc_z_pojemnosci_2], 'Is_Anomaly': [is_anomaly_2]
})[features]

wycena_2_pln = np.exp(model.predict(input_data_2)[0])

print(f"\n==============================================================================")
print(f"--- REKORD 2: BMW Seria 5 530i M Sport (Rocznik 2019) ---")
print(f"==============================================================================")
print(f"-> Cena z ogłoszenia:  {cena_2:,.2f} PLN")
print(f"-> Wycena modelu:      {wycena_2_pln:,.2f} PLN")
roznica_2 = ((wycena_2_pln - cena_2) / cena_2) * 100
print(f"-> Różnica:            {roznica_2:.2f}% (Model uważa za: {'OKAZJĘ' if roznica_2 > 0 else 'ZAWYŻONE'})")
print(f"-> Status anomalii IF: {'Wykryto odchylenie (Outlier)' if is_anomaly_2 == 1 else 'Normalny profil rynkowy'}")

bylo_w_train_2 = X_train[(X_train['Wiek'] == wiek_2) & (X_train['Przebieg'] == przebieg_2) & (X_train['Moc'] == moc_2)]
if len(bylo_w_train_2) > 0:
    print(f"-> [WERYFIKACJA]: Wykryto to auto w zbiorze TRENINGOWYM ({len(bylo_w_train_2)} szt.).")
else:
    print(f"-> [WERYFIKACJA]: Brak w zbiorze treningowym. Predykcja jest w 100% czysta!")
print("==============================================================================")
