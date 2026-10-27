# 🚗 BMW Used Car Valuation & Price Prediction (CatBoost & Isolation Forest)

Zaawansowany projekt uczenia maszynowego dedykowany automatycznej wycenie samochodów marki BMW na podstawie danych rynkowych z serwisu Otomoto. Model analizuje cechy techniczne i eksploatacyjne pojazdów, oceniając, czy cena ogłoszeniowa stanowi okazję rynkową, czy wartość zawyżoną.

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
