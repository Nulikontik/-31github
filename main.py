
def get_data(page_num):
    url = f"https://salexy.kg/osh/nedvizhimost/uchastki?page={page_num}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        return None

def parse_data(data):
    bs = BeautifulSoup(data, 'html.parser')
    prices = bs.find_all('div', class_='price')
    addresses = bs.find_all('div', class_='properties')  # Fixed the typo here
    data_list = []
    for price, address in zip(prices, addresses):
        price_text = price.get_text(strip=True)
        address_text = address.get('title', '')
        data_list.append((price_text, address_text))
    return data_list

page_numbers = [1, 2]
all_data = []

for page_num in page_numbers:
    data = get_data(page_num)
    if data:
        parsed_data = parse_data(data)
        all_data.extend(parsed_data)

for price, address in all_data:
    print("Price:", price)
    print("Address:", address)
    print()
