# 🚗 BMW Used Car Valuation & Price Prediction (CatBoost & Isolation Forest)

Zaawansowany projekt uczenia maszynowego dedykowany automatycznej wycenie samochodów marki BMW na podstawie danych rynkowych z serwisu Otomoto. Model analizuje cechy techniczne i eksploatacyjne pojazdów, oceniając, czy cena ogłoszeniowa stanowi okazję rynkową, czy wartość zawyżoną. Dane zostały zebrane autorskim scraperem z serwisu otomoto.

---

## 📊 Wyniki i Skuteczność Modelu

Model **CatBoost Regressor** wytrenowany na zalogarytmowanych cenach osiągnął następujące wyniki na zbiorze testowym:

* **R² Score:** `0.92` (model objaśnia 92% wariancji cen samochodów)
* **MAPE:** `9.4%` (średni błąd względny predykcji wynosi poniżej 10%)

> **Uwaga metodologiczna:** Ze względu na szeroki zakres cenowy w zbiorze (dane obejmują auta o wartości od 90 000 PLN do 500 000 PLN), główną metryką ewaluacyjną jest błąd względny (MAPE) oraz współczynnik $R^2$.

---

## 🛠️ Architektura i Zaawansowane Techniki

* **Feature Engineering:** Tworzenie wskaźników domenowych (`Przebieg_Roczny`, `Moc_Z_Pojemnosci`), kategoryzacja linii modelowych (`Seria_Main`) oraz automatyczna detekcja wersji sportowych (`Is_M_Power`).
* **Detekcja Anomalii (Isolation Forest):** Wykorzystanie modelu nienadzorowanego do automatycznego wykrywania i flagowania nietypowych profili pojazdów (`Is_Anomaly`).
* **Transformacja Zmiennej Celu:** Zastosowanie logarytmu naturalnego $log(y)$ w celu ustabilizowania wariancji cen i lepszego dopasowania modelu.
* **Weryfikacja Wycieku Danych:** Rygorystyczny podział próby oraz automatyczna kontrola braku obecności próbek testowych w zbiorze treningowym.

---

## 💻 Stack Technologiczny

* **Język:** Python
* **Machine Learning:** CatBoost, Scikit-learn (Isolation Forest, train_test_split, metryki)
* **Przetwarzanie Danych:** Pandas, NumPy

  ## 📊 Wyniki Ewaluacji Modelu

Model został przetestowany na wydzielonej próbie testowej (20% danych) z zachowaniem pełnej rygorystyki braku wycieku danych:

* **R² Score:** `0.9117` (model objaśnia 91.17% wariancji cen samochodów)
* **MAPE:** `9.44%` (średni błąd względny predykcji wynosi poniżej 10%)
* **MAE:** `18 616 PLN`

### 🎯 Dystrybucja Błędów Predykcji (Wskaźnik Dokładności Wyceny)

| Tolerancja Błędu (%) | Udział Predykcji w Przedziale | Interpretacja |
| :---: | :---: | :--- |
| **<= 5%** | **37.35%** | Niemal idealne odzwierciedlenie wartości rynkowej |
| **<= 10%** | **65.53%** | Dwie trzecie aut wycenionych z bardzo wysoką precyzją |
| **<= 15%** | **82.49%** | Zdecydowana większość próby w wąskim marginesie błędu |
| **<= 20%** | **91.52%** | Model poprawnie szacuje wartość ponad 9 na 10 pojazdów |
| **<= 30%** | **97.26%** | Wyeliminowanie rażących pomyłek wyceny |

## 🔍 Interpretowalność Modelu i Analiza Ważności Cech (Explainable AI - XAI)
> Analiza mechanizmu podejmowania decyzji przez model (otwarcie „czarnej skrzynki”) przy użyciu wartości **SHAP** oraz **CatBoost Feature Importance**. Wykresy przedstawiają cechy, na których model bazuje podczas generowania predykcji.

<img width="100%" alt="CatBoost Feature Importance" src="https://github.com/user-attachments/assets/9feee5c9-92d5-4c4e-a4b7-ff74a798859c" />

<img width="100%" alt="SHAP Summary Plot" src="https://github.com/user-attachments/assets/6003aba2-62f5-4ca1-ad94-543c0b63c37b" />

> 🚧 **Status projektu:** *Work in Progress (WIP)*  
> Główny pipeline Machine Learning (CatBoost, Isolation Forest, Feature Engineering oraz ewaluacja $R^2 = 0.91$, MAPE $= 9.4\%$) jest w pełni funkcjonalny i przetestowany.  
> **W trakcie realizacji i planowane rozszerzenia:**
> * Szczegółowa, pogłębiona interpretacja i opisy wykresów **SHAP** oraz **Feature Importance**.
> * Wdrożenie interfejsu **API** (FastAPI/Flask) do predykcji cen w czasie rzeczywistym.
> * Potencjalne dalsze usprawnienia (np. dodanie nowych cech z treści ogłoszeń lub konteneryzacja za pomocą Docker).

## Autor MIchał Proskurnicki 3 rok informatyki i ekonometrii UEP.



