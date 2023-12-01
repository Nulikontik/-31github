!pip install selenium
!apt-get update
!apt install -y chromium-chromedriver
!cp /usr/lib/chromium-browser/chromedriver /usr/bin
!pip install selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time
import pandas as pd

def scroll_down(driver, num_scrolls):
    for _ in range(num_scrolls):
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(2)

def parse_data(data):
    bs = BeautifulSoup(data, 'html.parser')
    addresses = bs.find_all('div', class_='ci9YC P3xSS')
    prices = bs.find_all('span', class_='eypL8 uwvkD')
    data_list = []
    for price, address in zip(prices, addresses):
        address_text = address.get('title', '')
        price_text = price.get_text(strip=True)
        data_list.append((address_text, price_text))
    return data_list


options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)


url = "https://msk.etagi.com/zastr/"
driver.get(url)


scroll_down(driver, num_scrolls=3)


data = driver.page_source


driver.quit()


parsed_data = parse_data(data)


df = pd.DataFrame(parsed_data, columns=['Адрес', 'Цена'])


print(df)


df.to_excel('output_data.xlsx', index=False)
