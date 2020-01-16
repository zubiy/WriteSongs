#нейронная сеть которая пишет текст песни. Обучающая выборка тексты группы "Руки Вверх"
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import cyrtranslit
from textgenrnn import textgenrnn

URL = 'http://txtmusic.ru/index.php?s=%D0%F3%EA%E8+%C2%E2%E5%F0%F5%21'

page = urllib.request.urlopen(URL) 
soup = BeautifulSoup(page)

link = soup.body.findAll('li') 
URLS = ['http://txtmusic.ru/'+li.a.get('href') for li in link] 
df = pd.DataFrame(columns=['name', 'text'])
list_of_names = []
list_of_text = []
for URL in URLS:
    page = urllib.request.urlopen(URL)
    soup = BeautifulSoup(page)
    name = soup.body.findAll('article') 
    article = soup.body.findAll('p') 
    text = str(article[0]).split('\n')
    text = str(text).split('<br/>')
    text = [t for t in text if t!='']
    del text[0]
    del text[-1]
    text = " ".join(text)

    name= str(name[0].h1).split(" - ")[1].rstrip("</h1>")
    list_of_text.append(text)
    list_of_names.append(name)

df.name = list_of_names
df.text = list_of_text

df.to_csv('songs.csv')
df = pd.read_csv('songs.csv')
df = df[['name','text']]
df.text = df.text.apply(lambda x: cyrtranslit.to_latin(x, 'ru'))
df.text.to_csv('trans.csv')

textgen = textgenrnn()
textgen.train_from_file('trans.csv', num_epochs=1)

textgen_2 = textgenrnn('textgenrnn_weights.hdf5')
textgen_2.generate(3, temperature=1.0)
textgen_2.generate_to_file('lyrics.txt')

for i in open('lyrics.txt'):
    s = cyrtranslit.to_cyrillic(i)
    print(s)
