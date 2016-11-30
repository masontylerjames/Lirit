from bs4 import BeautifulSoup
import requests
import urllib
import os

start_url = 'http://www.piano-midi.de/midi_files.htm'
domain = 'http://www.piano-midi.de/'
savelocation = '/home/tyler/Documents/Galvanize_DSI/Project/Lirit/data/train/'
# savelocation = '../data/train/'


def make_soup(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup

if __name__ == '__main__':
    soup = make_soup(start_url)
    navi = soup.select('.navi')
    hrefs = [tag.attrs['href'] for tag in navi]
    hrefs = hrefs[1:26]
    soups = {}
    for href in hrefs:
        soups[href] = make_soup(domain + href)
    for key, value in soups.iteritems():
        navis = value.select('.navi')
        links = [tag.attrs['href'] for tag in navis]
        midi_links = [
            domain + link for link in links if '.mid' in link and 'midis/' in link]
        for link in midi_links:
            author = link.split('/')[-2]
            title = link.split('/')[-1]
            filedir = savelocation + author
            filename = filedir + '/' + title
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            print 'downloading {} from {}'.format(title, link)
            urllib.urlretrieve(link, filename)
    pass
