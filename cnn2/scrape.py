import os, requests, string
from bs4 import BeautifulSoup

root = "http://cnn.com"


def remove_punctuation(word):
    if word:
        return ''.join([ch for ch in word if ch not in string.punctuation])
    else:
        return ''

def get_dict(section, soup):
    "Return a dictionary of article titles and their urls for a given section."
    
    a_dict = {}

    if section == "Money":
        articles = soup.find_all("a", class_ = "homepage-stack-hed", limit = 10)
        for a in articles:
            href = a["href"]
            title = a.getText()
        
            if href[0] == '/':
                href = "http://money.cnn.com" + href

            a_dict[title] = href

    elif section == "Sports":
        articles = soup.find_all("a", class_ = "title", limit = 10)

        for a in articles:
            href = a.attrs["href"]
            title = a.find("span", class_ = "title").getText()
            
            if "bleacherreport" in href:
                a_dict[title] = href

    else:
        articles = soup.find_all('h3', class_ = "cd__headline", limit = 10)

        for a in articles:
            href = a.find("a").attrs["href"]
            title = a.find("span").contents[0]

            if href[0] == '/':
                href = root + href

            a_dict[title] = href

    return a_dict


def parse(section, url):

    section_page = requests.get(url).content
    section_soup = BeautifulSoup(section_page, 'html.parser')

    a_dict = get_dict(section, section_soup)

    for title, href in a_dict.items():

        page = requests.get(href).content
        soup = BeautifulSoup(page, 'html.parser')

        if 'money.cnn' in href:
            try:
                story = soup.find("div", id = "storytext").find_all(["p", "blockquote"])
            except AttributeError:
                pass

        elif 'bleacherreport' in href:
            try:
                story = soup.find("div", class_ = "article_body").find_all(["p", "blockquote"])
            except AttributeError:
                pass
        else:
            try:
                story = soup.find_all(class_ = "zn-body__paragraph")
            except AttributeError:
                pass

        paragraphs = [ p.getText() for p in story]
        full_text = '\n'.join(paragraphs)

        if len(full_text) > 250 and title is not None:
            try:
                file_name = '-'.join(map(remove_punctuation, title.lower().split())) 

                path = os.path.join('articles', section)
                if not os.path.isdir(path):
                    os.makedirs(path)

                print("Saving article as {0}".format(file_name))
                with open(os.path.join(path, file_name), "w") as f:
                    f.write(full_text)
            except TypeError:
                pass


if __name__ == '__main__':

    home_page = requests.get(root).content

    soup = BeautifulSoup(home_page, 'html.parser')

    links = soup.find_all("a", class_ = "nav-menu-links__link")
    sections = {link.contents[0] : link.attrs["href"] for link in links}

    for section, ext in sections.items():

        if section == 'Video':
            continue

        if '//' in ext:
            parse(section, ext.replace('//', 'http://'))
        else:
            parse(section, root + ext)


