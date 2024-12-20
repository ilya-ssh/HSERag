import os
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track
import unicodedata
import string

console = Console()

def download_file(url, file_name):
    response = requests.get(url, stream=True)
    with open(file_name, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

def normalize_text(text):
    cyrillic_chars = ''.join(chr(i) for i in range(0x0400, 0x0500))#???
    allowed_chars = string.ascii_letters + string.digits + string.punctuation + ' \n' + cyrillic_chars
    normalized_text = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in normalized_text if c in allowed_chars)

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    normalized_text = normalize_text(text)
    with open(file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(normalized_text)

links = [
    'https://www.hse.ru/data/xf/414/079/1119/%D0%9F%D0%92%D0%A0%D0%9E%202024.docx',
    'https://www.hse.ru/data/xf/536/300/2020/7.18.1-01_300523-19%20%D0%98%D1%82%D0%BE%D0%B3%D0%BE%D0%B2%D1%8B%D0%B9%20%D0%B4%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82.docx',
    'https://www.hse.ru/data/xf/259/437/1420/6.18.1-01_110823-1%20%D0%98%D1%82%D0%BE%D0%B3%D0%BE%D0%B2%D1%8B%D0%B9%20%D0%B4%D0%BE%D0%BA%D1%83%D0%BC%D0%B5%D0%BD%D1%82.docx',
    'https://www.hse.ru/data/xf/221/433/1420/%D0%9F%D0%BE%D0%BB%D0%BE%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5%20%D0%BE%20%D0%9F%D0%9F.docx'
]

console.print("Downloading files", style="magenta")
for i in track(range(len(links)), description="Downloading DOCX files..."):
    url = links[i]
    file_name = f'regfile_{i+1}.docx'
    download_file(url, file_name)

console.print("Downloading handbook files", style="magenta")
url = "https://www.hse.ru/studyspravka/handbook/studyspravka_std"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
info_items = soup.find_all(class_="info-board__item")

handbook_links = []
for item in info_items:
    link_tag = item.find('a') 
    if link_tag and link_tag['href']:
        handbook_links.append(link_tag['href'])

for link in track(handbook_links, description="handbook texts..."):
    link_response = requests.get(link)
    link_soup = BeautifulSoup(link_response.content, 'html.parser')
    text_element = link_soup.find(class_="post__text")
    
    if text_element:
        text = text_element.get_text(strip=True)
        normalized_text = normalize_text(text)
        file_name = link.split('/')[-1] if link.split('/')[-1] else "document"
        file_name = file_name.replace('.html', '') + ".txt"
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(normalized_text)

console.print("Normalizing...", style="magenta")
txt_files = [f for f in os.listdir('.') if f.endswith('.txt')]

for file_path in track(txt_files, description="Processing TXT files..."):
    process_file(file_path)
    console.print(f" {file_path}", style="green")

console.print("NORMALIZED", style="magenta")
