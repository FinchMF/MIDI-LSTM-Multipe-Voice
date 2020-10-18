
import os
import re
import requests
from bs4 import BeautifulSoup
import wget


params = {

    'root': 'http://www.piano-midi.de/',
    'midi_page': 'midi_files.htm',
    'midi_world': 'https://www.midiworld.com/',
    'composer_page': 'composers.htm',
    'output_data': 'data/compose/'

}

def download_midi(composer, params):

    html_page = requests.get(params['root'] + f'{composer}')
    midi_soup = BeautifulSoup(html_page.content, 'html.parser')

    for table in midi_soup.findAll('table', 'midi'):
        for link in table.findAll('a'):

            if link.get('href')[-3:] == 'mid':
                print('\n [i] File Extension:' + link.get('href')[-3:])
                fname = link.get('href')
                wget.download(params['root'] + f'{fname}', out=params['output_data']+ f'{composer[:-4]}')

    return None

def scrape_piano_midi(params):

    html_page = requests.get(params['root'] + params['midi_page'])
    composer_soup = BeautifulSoup(html_page.content, 'html.parser')
    composer_table = composer_soup.find('table', 'midi')

    for link in composer_table.findAll('a'):

        print('\n[i] Finding Works by: \n'+ link.get('href')[:-4])
        os.mkdir(params['output_data'] + link.get('href')[:-4])

        download_midi(link.get('href'), params)

    return None


def scrape_midi_world(params):

    html_page = requests.get(params['midi_world'] + params['composer_page'])
    composer_page = BeautifulSoup(html_page.content, 'html.parser')

    find_composer = composer_page.findAll('a')

    for composer_link in find_composer:

        match = re.search('.htm$', str(composer_link.get('href')))
        
        if match:

            print('\n[i] Fining Works by: \n' + str(composer_link.get('href')[:-4]))
            try:
                os.mkdir('data/compose/' + str(composer_link.get('href')[:-4]))
            except FileExistsError:
                continue

            url = params['midi_world'] + composer_link.get('href')

            composer_html = requests.get(url)
            composer_midi_soup = BeautifulSoup(composer_html.content, 'html.parser')
            composer_midi_links = composer_midi_soup.findAll('a', attrs={'href': re.compile('^https://')})

            for link in composer_midi_links:

                if link.get('href')[-3:] == 'mid':
                    print('[+] Downloading ' + link.get('href'))
                    wget.download(link.get('href'), out=params['output_data'] + composer_link.get('href')[:-4])

    return None

        

class Piano_Crawler:

    def __init__(self, params=params):

        self.params = params

    def scrape(self):

        print('[+] Retrieving Piano MIDI Data')

        scrape_piano_midi(self.params)
        scrape_midi_world(self.params)

        print('[+] Retrieved All Piano MIDI Data')

        return None