from bs4 import BeautifulSoup
!pip3 install bs4
import requests
import openpyxl
url = 'https://lalafo.kg/osh/zemelnye-uchastki/prodazha-zemli'
response = requests.get(url)
print(response)
!pip install openpyxl
!pip install pandas
!pip install beautifulsoup4

import openpyxl
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://lalafo.kg/osh/zemelnye-uchastki/prodazha-zemli'
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    a_tags = soup.find_all('a')
    p_tags = soup.find_all('p')

    data = []

    for a_tag, p_tag in zip(a_tags, p_tags):
        title = a_tag.get_text()
        price = p_tag.get_text()
        data.append({'Title': title, 'Price': price})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to an Excel file
    df.to_excel('parsed_data.xlsx', index=False)
    print("Data saved to 'parsed_data.xlsx'.")
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

url = 'https://lalafo.kg/osh/zemelnye-uchastki/prodazha-zemli'
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    a_tags = soup.find_all('a')
    data_links = []
    for a_tag in a_tags:
        a_href = a_tag.get('href')
        if a_href and 'user' not in a_href:
            full_link = 'https://lalafo.kg' + a_href
            if full_link not in data_links:
                data_links.append(full_link)
    print(data_links)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")



def extract_description_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        description_wrap = soup.find("div", class_="description__wrap")
        prices = soup.find_all("span", class_="heading price")

        # Check if there is at least one price associated with the description
        if prices:
            price_text = ", ".join([price.text for price in prices])
        else:
            price_text = "No price found"

        if description_wrap:
            p_tags = description_wrap.find_all("p")
            description_text = " ".join([p.text for p in p_tags])
        else:
            description_text = "Description not found"

        return {"Link": url, "Price": price_text, "Description": description_text}
    except requests.exceptions.RequestException as e:
        return {"Link": url, "Price": "Error", "Description": str(e)}

data = []

for link in data_links:
    description_data = extract_description_text(link)
    data.append(description_data)

df = pd.DataFrame(data)
df.to_excel("parsed_data.xlsx", index=False)
print("Data saved to 'parsed_data.xlsx'.")
