import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.ok:
            return response.content
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error while fetching the page: {e}")
        return None

def extract_data_from_page(content):
    soup = BeautifulSoup(content, 'html.parser')
    a_tags = soup.find_all('a')
    p_tags = soup.find_all('p')

    data = []

    for a_tag, p_tag in zip(a_tags, p_tags):
        title = a_tag.get_text()
        price = p_tag.get_text()
        data.append({'Title': title, 'Price': price})

    return data

def save_data_to_excel(data, filename):
    df = pd.DataFrame(data)
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.save()

def extract_description_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        description_wrap = soup.find("div", class_="description__wrap")
        prices = soup.find_all("span", class_="heading price")

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

def main():
    url = 'https://lalafo.kg/osh/zemelnye-uchastki/prodazha-zemli'

    page_content = get_page_content(url)
    if page_content:
        data = extract_data_from_page(page_content)
        save_data_to_excel(data, 'parsed_data.xlsx')
        print("Data saved to 'parsed_data.xlsx'.")
    else:
        print(f"Failed to retrieve the page at {url}")

    # Extract and save additional data from links
    response_links = get_page_content(url)
    if response_links:
        soup_links = BeautifulSoup(response_links, 'html.parser')
        a_tags_links = soup_links.find_all('a')
        data_links = []

        for a_tag in a_tags_links:
            a_href = a_tag.get('href')
            if a_href and 'user' not in a_href:
                full_link = 'https://lalafo.kg' + a_href
                if full_link not in data_links:
                    data_links.append(full_link)

        additional_data = []
        for link in data_links:
            description_data = extract_description_text(link)
            additional_data.append(description_data)

        save_data_to_excel(additional_data, 'additional_data.xlsx')
        print("Additional data saved to 'additional_data.xlsx'.")

if __name__ == "__main__":
    main()
