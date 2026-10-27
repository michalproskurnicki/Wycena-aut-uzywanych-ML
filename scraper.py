import os
import time
import pandas as pd
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def clean_text(text):
    if text:
        return text.strip().replace('\xa0', ' ')
    return ""

def save_data(cars_data):
    df = pd.DataFrame(cars_data)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop):
        desktop = os.path.join(os.path.expanduser("~"), "Pulpit")
        
    folder_path = os.path.join(desktop, "bmw_projekt")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    output_filepath = os.path.join(folder_path, "otomoto_bmw_pelna_specyfikacja.xlsx")
    try:
        df.to_excel(output_filepath, index=False)
        print(f"\n[SUKCES] Zapisano plik: {output_filepath}")
        return output_filepath
    except Exception as e:
        print(f"\n[BŁĄD ZAPISU] {e}")
        return None

def scrape_otomoto_full_flow():
    base_url = "https://www.otomoto.pl/osobowe/bmw"
    cars_data = []
    CURRENT_PAGE = 450

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=['--disable-blink-features=AutomationControlled', '--start-maximized']
        )
        
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1920, "height": 1080},
            locale="pl-PL"
        )
        
        page = context.new_page()
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

        while True:
            target_url = f"{base_url}?page={CURRENT_PAGE}"
            print(f"\n=== ANALIZA STRONY NR: {CURRENT_PAGE} ===")
            page.goto(target_url, wait_until="load")

            try:
                page.wait_for_selector('button#onetrust-accept-btn-handler', timeout=4000)
                page.click('button#onetrust-accept-btn-handler')
                time.sleep(1)
            except Exception: pass
            
            try:
                page.wait_for_selector('article[data-testid="listing-ad"], [data-cy="search-results-list"] article', timeout=10000)
                for i in range(1, 5):
                    page.evaluate(f"window.scrollTo(0, {i * 1000});")
                    time.sleep(0.5)
            except Exception:
                time.sleep(10)

            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')
            ads = soup.find_all('article', attrs={'data-testid': 'listing-ad'}) or soup.find_all('article')
            
            if not ads:
                print("Brak ogłoszeń. Kończę.")
                break
            
            page_listings = []
            for ad in ads:
                link_tag = ad.find('a', href=True)
                if not link_tag or "oferta/" not in link_tag.get('href'): continue
                
                car_info = {
                    "Marka": "BMW", "Seria/Model": "Brak danych", "Cena": "Brak danych",
                    "Rocznik": "Brak danych", "Typ sprzedawcy": "Nieokreślony", "Moc": "Brak danych",
                    "Pojemność": "Brak danych", "Typ paliwa": "Brak danych", "Typ skrzyni": "Brak danych",
                    "Typ nadwozia": "Brak danych", "Generacja": "Brak danych", "Przebieg": "Brak danych",
                    "Kraj pochodzenia": "Brak danych", "Serwisowany w ASO": "Brak danych",
                    "Stan": "Brak danych", "Zarejestrowany w Polsce": "Brak danych",
                    "Link": link_tag.get('href')   
                }
                title_tag = ad.find(['h1', 'h2', 'p'], attrs={'data-testid': 'ad-title'}) or ad.find('h2') or ad.find('h1')
                if title_tag:
                    pelny_tytul = clean_text(title_tag.text)
                    car_info["Seria/Model"] = pelny_tytul.replace("BMW ", "", 1) if pelny_tytul.startswith("BMW") else pelny_tytul
                price_section = ad.find(attrs={'data-testid': 'ad-price'}) or ad.find('h3')
                if price_section: car_info["Cena"] = clean_text(price_section.text)
                page_listings.append(car_info)

            print(f"-> Zidentyfikowano {len(page_listings)} ofert.")
            for car in page_listings:
                detail_page = context.new_page()
                try:
                    detail_page.goto(car["Link"], wait_until="domcontentloaded")
                    detail_page.wait_for_selector('main, div[data-testid="content-details-section"]', timeout=8000)
                    
                    try: detail_page.click('button:has-text("Rozwiń wszystkie")', timeout=1000)
                    except: pass
                    
                    detail_soup = BeautifulSoup(detail_page.content(), 'html.parser')
                    
                    # Typ sprzedawcy
                    for element in detail_soup.find_all(['p', 'span', 'div', 'h3']):
                        txt = clean_text(element.text)
                        if "Osoba prywatna" in txt: car["Typ sprzedawcy"] = "Osoba prywatna"; break
                        elif "Firma" in txt or "Dealer" in txt: car["Typ sprzedawcy"] = "Firma"; break
                    
                    # Parametry
                    all_labels = detail_soup.find_all(['p', 'span', 'div'])
                    for label_tag in all_labels:
                        lbl = clean_text(label_tag.text)
                        target_fields = [
                            "Moc", "Pojemność skokowa", "Typ nadwozia", "Generacja", 
                            "Rodzaj paliwa", "Skrzynia biegów", "Rok produkcji", "Przebieg",
                            "Kraj pochodzenia", "Serwisowany w ASO", "Stan", "Zarejestrowany w Polsce"
                        ]
                        if lbl in target_fields:
                            parent = label_tag.find_parent()
                            if parent:
                                try:
                                    val = clean_text(parent.find_all(['p', 'span', 'a'])[1].text)
                                    mapping = {
                                        "Moc": "Moc", "Pojemność skokowa": "Pojemność", "Typ nadwozia": "Typ nadwozia", 
                                        "Generacja": "Generacja", "Rodzaj paliwa": "Typ paliwa", "Skrzynia biegów": "Typ skrzyni",
                                        "Rok produkcji": "Rocznik", "Przebieg": "Przebieg",
                                        "Kraj pochodzenia": "Kraj pochodzenia", "Serwisowany w ASO": "Serwisowany w ASO",
                                        "Stan": "Stan", "Zarejestrowany w Polsce": "Zarejestrowany w Polsce"
                                    }
                                    car[mapping[lbl]] = val
                                except: pass
                except: pass
                finally:
                    detail_page.close()
                    time.sleep(1.2)
                
                cars_data.append(car)
                save_data(cars_data)
            
            CURRENT_PAGE += 1
        browser.close()

if __name__ == "__main__":
    scrape_otomoto_full_flow()
