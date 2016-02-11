import requests
from bs4 import BeautifulSoup

root = "http://cnn.com"

home_page = requests.get(root).content
soup = BeautifulSoup(home_page, 'html.parser')

urls = [ link.get("href") for link in soup.select("h2.cn__title ~ li a")[:10]]
top_stories = [ root + url if url[0] == '/' else url for url in urls]


stories = []
for url in top_stories:
    page = requests.get(url).content
    page_soup = BeautifulSoup(page, 'html.parser')

    # CNN's money section has different page structure
    if "money.cnn" in url:
        raw_story = page_soup.find("div", id = "storytext").find_all("p")
    else:
        raw_story = page_soup.find_all("p", class_ = "zn-body__paragraph")[1: -1]

    stories.append( ' '.join([line.get_text() for line in raw_story]))
    

# Save backup text to file
with open("stories.txt", "w") as f:
    for story in stories:
        f.write(story + "\n")




