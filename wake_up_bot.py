import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

APP_URL = "https://carbonocircular.streamlit.app/"

options = webdriver.ChromeOptions()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

try:
    driver.get(APP_URL)
    WebDriverWait(driver, 30).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )
    time.sleep(5)

    # Tenta clicar no botão de despertar do Streamlit (texto padrão)
    try:
        wake_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(text(), 'Yes, get this app back up!')]")
            )
        )
        wake_button.click()
        print("✅ Botão de acordar clicado! Aguardando o app iniciar...")
        time.sleep(20)
    except:
        print("ℹ️ App já está acordado ou botão não encontrado.")

finally:
    driver.quit()
