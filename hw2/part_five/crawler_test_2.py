import requests
from bs4 import BeautifulSoup

def fetch_wikipedia_content(wikipedia_url, num_documents=10):
    document_contents = []
    response = requests.get(wikipedia_url)
    if response.status_code != 200:
        print(f"Failed to fetch the page. Status code: {response.status_code}")
        return
    wikipedia_html = response.text
    wikipedia_soup = BeautifulSoup(wikipedia_html, 'html.parser')
    wikipedia_links = wikipedia_soup.find_all('a', href=True)
    relevant_wikipedia_links = [link['href'] for link in wikipedia_links if link['href'].startswith('/wiki/')]
    for i, link in enumerate(relevant_wikipedia_links[:num_documents]):
        full_url = f"https://en.wikipedia.org{link}"
        link_response = requests.get(full_url)
        if link_response.status_code == 200:
            link_html_content = link_response.text
            document_contents.append(link_html_content)
    
    with open('output_data.csv', 'w+') as csv_file:
        csv_file.write('_,_,_,_')
        for document_content in document_contents:
            document_content = document_content.replace("\n", "")
            csv_file.write(f'_,_,{document_content}\n')


target_wikipedia_url = 'https://en.wikipedia.org/wiki/Main_Page'
fetch_wikipedia_content(target_wikipedia_url)
