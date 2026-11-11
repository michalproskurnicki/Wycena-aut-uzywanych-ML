import pandas as pd
import numpy as np
import re
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest

# ==============================================================================
# 1. WCZYTANIE I WSTĘPNE CZYSZCZENIE DANYCH
# ==============================================================================
file_path = 'C:/Users/PC/Desktop/bmw/dane_finalne.xlsx'  
df = pd.read_excel(file_path, dtype=str)
df.columns = df.columns.str.strip()

def clean_hard(x):
    if pd.isna(x) or x == 'nan': return None
    return ''.join([c for c in str(x) if c.isdigit()])

cols_to_clean = ['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg']
for col in cols_to_clean:
    df[col] = pd.to_numeric(df[col].apply(clean_hard), errors='coerce')

df = df.dropna(subset=['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg'])
df = df[(df['Cena'] >= 90000) & (df['Cena'] <= 1000000)]
df = df[(df['Moc'] > 50)]

mask_nowe = df['Stan'].astype(str).str.strip().str.lower().isin(['nowe', 'nowy'])
df = df[(df['Przebieg'] > 1000) | ((df['Przebieg'] <= 1000) & mask_nowe)]

identyfikatory_auta = ['Cena', 'Moc', 'Pojemność', 'Rocznik', 'Przebieg', 'Seria/Model']
df = df.drop_duplicates(subset=identyfikatory_auta).reset_index(drop=True)

# ==============================================================================
# 2. INŻYNIERIA CECH BEZWZGLĘDNYCH
# ==============================================================================
AKTUALNY_ROK = 2026
df['Wiek'] = AKTUALNY_ROK - df['Rocznik']
df['Przebieg_Roczny'] = df['Przebieg'] / (df['Wiek'] + 0.5)
df['Moc_Z_Pojemnosci'] = df['Moc'] / (df['Pojemność'] + 1)

def parse_seria_main(val):
    if pd.isna(val) or val == 'Brak danych': return 'Brak'
    val_str = str(val).strip()
    if val_str.lower().startswith('seria'):
        parts = val_str.split(' ')
        return ' '.join(parts[:2]) if len(parts) >= 2 else parts[0]
    match = re.match(r'^([A-Za-z]\d|\d{3}|M\d?)', val_str, re.IGNORECASE)
    return match.group(0).upper() if match else val_str.split(' ')[0]

def parse_is_m_power(val):
    if pd.isna(val): return 0
    pattern = r'(^M\d{1,2}\b)|(\bM\b)|(^M\d{3})'
    return 1 if re.search(pattern, str(val).strip(), re.IGNORECASE) else 0

df['Seria_Main'] = df['Seria/Model'].apply(parse_seria_main)
df['Is_M_Power'] = df['Seria/Model'].apply(parse_is_m_power)
df['Is_xDrive'] = df['Seria/Model'].astype(str).str.contains('xDrive', case=False).astype(int)
df['Is_Electric_Hybrid'] = df['Typ paliwa'].astype(str).str.contains('hybryda|elektryczny', case=False).astype(int)

cols_cat = ['Typ sprzedawcy', 'Kraj pochodzenia', 'Typ skrzyni', 'Typ nadwozia', 'Typ paliwa', 'Generacja', 'Seria/Model', 'Stan', 'Seria_Main']
for col in cols_cat:
    df[col] = df[col].fillna('Brak danych').astype(str)

features_base = [
    'Typ sprzedawcy', 'Kraj pochodzenia', 'Typ skrzyni', 'Typ nadwozia', 
    'Typ paliwa', 'Generacja', 'Seria/Model', 'Stan', 'Seria_Main', 
    'Wiek', 'Rocznik', 'Moc', 'Pojemność', 'Przebieg', 'Is_M_Power', 
    'Is_xDrive', 'Is_Electric_Hybrid', 'Przebieg_Roczny', 'Moc_Z_Pojemnosci'
]
cat_features = cols_cat

X = df[features_base].copy()
y = df['Cena'].values
y_log = np.log(y)

# ==============================================================================
# 3. WALIDACJA KRZYŻOWA (5-FOLD CV) - CATBOOST
# ==============================================================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
features_for_anomaly = ['Wiek', 'Moc', 'Pojemność', 'Przebieg', 'Przebieg_Roczny', 'Moc_Z_Pojemnosci']

metrics = {'Train_MAPE': [], 'Val_MAPE': [], 'Train_R2': [], 'Val_R2': [], 'Val_MAE': [], 'Val_RMSE': []}

print("==================================================")
print("  URUCHAMIAM WALIDACJĘ KRZYŻOWĄ: CATBOOST REGRESSOR")
print("==================================================")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_log), 1):
    X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_tr_log, y_val_log = y_log[train_idx], y_log[val_idx]
    y_tr_real, y_val_real = y[train_idx], y[val_idx]

    sredni_przebieg_rocznika = X_tr.groupby('Rocznik')['Przebieg'].mean().to_dict()
    glob_przebieg = X_tr['Przebieg'].mean()
    X_tr['Odchylenie_Przebiegu'] = X_tr['Przebieg'] - X_tr['Rocznik'].map(sredni_przebieg_rocznika).fillna(glob_przebieg)
    X_val['Odchylenie_Przebiegu'] = X_val['Przebieg'] - X_val['Rocznik'].map(sredni_przebieg_rocznika).fillna(glob_przebieg)

    srednia_moc_serii = X_tr.groupby('Seria_Main')['Moc'].mean().to_dict()
    glob_moc = X_tr['Moc'].mean()
    X_tr['Stosunek_Mocy_Do_Serii'] = X_tr['Moc'] / X_tr['Seria_Main'].map(srednia_moc_serii).fillna(glob_moc)
    X_val['Stosunek_Mocy_Do_Serii'] = X_val['Moc'] / X_val['Seria_Main'].map(srednia_moc_serii).fillna(glob_moc)

    q_low, q_high = np.percentile(y_tr_real, 1), np.percentile(y_tr_real, 99)
    mask_quantile = (y_tr_real >= q_low) & (y_tr_real <= q_high)

    iso = IsolationForest(contamination=0.02, random_state=42)
    iso.fit(X_tr[features_for_anomaly])
    mask_iso = (iso.predict(X_tr[features_for_anomaly]) == 1)

    final_tr_mask = mask_quantile & mask_iso
    features_final = features_base + ['Odchylenie_Przebiegu', 'Stosunek_Mocy_Do_Serii']

    X_tr_clean = X_tr.loc[final_tr_mask, features_final]
    y_tr_log_clean = y_tr_log[final_tr_mask]
    y_tr_real_clean = y_tr_real[final_tr_mask]

# Inicjalizacja modelu z optymalną regularyzacją anty-overfittingową
    cb = CatBoostRegressor(
        iterations=1500,           # Stabilna liczba drzew
        depth=6,                   # Bezpieczniejsza głębokość drzewa (zapobiega zapamiętywaniu)
        learning_rate=0.03,        # Wolne, precyzyjne dopasowywanie
        l2_leaf_reg=7.0,           # Podwyższona kara L2 w liściach
        random_strength=1.0,       # Wyższa losowość przy wybieraniu punktów podziału
        subsample=0.8,             # Każde drzewo widzi 80% losowych wierszy z treningu
        cat_features=cat_features,
        logging_level='Silent',
        random_seed=42
    )

    # Czyste dopasowanie WYŁĄCZNIE na zbiorze treningowym fałdy (brak Data Leakage)
    cb.fit(
        X_tr_clean,
        y_tr_log_clean
    )

    pred_tr_real = np.exp(cb.predict(X_tr_clean))
    pred_val_real = np.exp(cb.predict(X_val[features_final]))

    metrics['Train_MAPE'].append(mean_absolute_percentage_error(y_tr_real_clean, pred_tr_real))
    metrics['Val_MAPE'].append(mean_absolute_percentage_error(y_val_real, pred_val_real))
    metrics['Train_R2'].append(r2_score(y_tr_real_clean, pred_tr_real))
    metrics['Val_R2'].append(r2_score(y_val_real, pred_val_real))
    metrics['Val_MAE'].append(mean_absolute_error(y_val_real, pred_val_real))
    metrics['Val_RMSE'].append(np.sqrt(mean_squared_error(y_val_real, pred_val_real)))

    print(f"Fałda {fold}/5 | Train MAPE: {metrics['Train_MAPE'][-1]:.2%} | Val MAPE: {metrics['Val_MAPE'][-1]:.2%} | Val R2: {metrics['Val_R2'][-1]:.4f}")

print("\n--- PODSUMOWANIE CATBOOST ---")
print(f"Średni Train MAPE: {np.mean(metrics['Train_MAPE']):.2%}")
print(f"Średni Val MAPE:   {np.mean(metrics['Val_MAPE']):.2%}")
print(f"Średni Train R2:   {np.mean(metrics['Train_R2']):.4f}")
print(f"Średni Val R2:     {np.mean(metrics['Val_R2']):.4f}")
print(f"Średni Val MAE:    {np.mean(metrics['Val_MAE']):,.2f} PLN")
print(f"Średni Val RMSE:   {np.mean(metrics['Val_RMSE']):,.2f} PLN")
