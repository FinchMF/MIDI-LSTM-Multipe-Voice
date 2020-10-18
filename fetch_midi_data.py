
from utils import params  
from utils import midi_crawler 
import build as b

def fetch():

    crawler = midi_crawler.Piano_Crawler()
    crawler.scrape()

    return None


def build():

    param = params.set_params()
    b.build_directories(param['run_folder'])

    return None


def fetch_and_build():
    
    build()
    fetch()
    
    return print('[i] Directories Built - Data Retrieved..')

if __name__ == '__main__':

    fetch_and_build()

    