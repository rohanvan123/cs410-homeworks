import os
import csv
import requests
from bs4 import BeautifulSoup

base_url = "https://en.wikipedia.org/wiki/"

download_directory = "downloaded_pages"
os.makedirs(download_directory, exist_ok=True)

# Create a CSV file to store the extracted data
csv_file_path = "documents.csv"
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Title', 'Description'])

    for i in range(1, 11):
        article_url = f"{base_url}Article_{i}"  
        response = requests.get(article_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title and text content
            title = soup.title.text if soup.title else ''
            text_content = soup.get_text(separator='\n')  # Use '\n' to separate lines

            # Save to CSV
            csv_writer.writerow([title, text_content])

            # Save HTML file
            file_path = os.path.join(download_directory, f"page_{i}.html")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(str(soup))

            print(f"Downloaded {article_url} to {file_path}")
        else:
            print(f"Failed to fetch {article_url}. Status code: {response.status_code}")

print(f"CSV file created: {csv_file_path}")
