import pandas as pd
import re
import urllib.parse
import unidecode
unicodeStrings = {
                "u2013" :	"–",
                "u2014" :	"—",
                "u2015" :	"―",
                "u2017" :	"‗",
                "u2018" :	"‘",
                "u2019" :	"’",
                "u201A" :	"‚",
                "u201B" :	"‛",
                "u201C" :	"“",
                "u201D" :	"”",
                "u201E" :	"„",
                "u2020" :	"†",
                "u2021" :	"‡",
                "u2022" :	"•",
                "u2026" :	"…",
                "u2030" :	"‰",
                "u2032" :	"′",
                "u2033" :	"″",
                "u2039" :	"‹",
                "u203A" :	"›",
                "u203C" :	"‼",
                "u203E" :	"‾",
                "u2044" :	"⁄"
                               }
def remove_url(txt):

    try:
        #txt = urllib.parse.unquote(txt)
        #txt = unidecode.unidecode(txt)
        # for key in unicodeStrings:
        #     if key in txt:
        #         txt = txt.replace(key, unicodeStrings[key])
        txt = " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

    except:
        txt = txt

    return txt


raw = pd.read_excel('historical_data/historical_tweets_2016-09-01_2016-10-01.xlsx')

processed = pd.DataFrame()
processed['content'] = raw['content']
processed['content_new'] = raw['content']

for idx, row in processed.iterrows():
       nowContent = remove_url(row['content_new'])
       #try:
           #nowContent = urllib.parse.unquote(nowContent)
       processed.at[idx, 'content_new'] = nowContent

print(processed.head(50).to_string())


