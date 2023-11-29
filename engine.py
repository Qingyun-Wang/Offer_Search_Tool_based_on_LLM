import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from fuzzywuzzy import fuzz

# Function to check and download necessary NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

# Call the function to ensure resources are downloaded
download_nltk_resources()


def input_prep(txt,stop_words):
    txt = re.sub(r'\$?\d+', '', txt)
    txt = txt.replace("'","").replace("$","").replace('-'," ")

    words = word_tokenize(txt)
    # words =[word.strip(u"\u2122").strip(u'\u0256') for word in words]
    filtered_text = [word for word in words if (not word.lower() in stop_words) and (word.isalpha())]
    tagged = pos_tag(filtered_text)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return " ".join(nouns)
def rule_prediction(row, txt):
    score = fuzz.partial_ratio(txt.upper(),str(row['training_str']).upper())
    return score

def run_engine(df, input_str, num_results=20):
    
    sw = set(stopwords.words('english'))
    sw.update({"buy", "spend", "select", 'varieties', 'sizes', 'ounce', 'count', 'liter'})
    sw.remove('any')
    
    txt = input_prep(input_str,sw)

    df['pred_score'] = df.apply(lambda row: rule_prediction(row, txt),axis=1)
    results = df.sort_values(by ='pred_score',ascending=False)[['OFFER',"pred_score"]].head(num_results)

    print(results[['OFFER', 'pred_score']].to_string(index=False))