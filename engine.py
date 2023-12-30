import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from fuzzywuzzy import fuzz
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import BertTokenizer, BertModel


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

def reshape_to_2d(series):
    # Convert the series to a NumPy array
    array_1d = np.array(series)
    
    # Reshape to a 2D array (1 row and N columns)
    array_2d = array_1d.reshape(1, -1)
    
    return array_2d

def compute_similarity(x, y):
    return cosine_similarity(x, y)[0][0]

def get_embedding(text):
    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize the input text and prepare it as input for the model
    encoded_input = tokenizer(text, return_tensors='pt')

    # Get the embeddings
    with torch.no_grad():
        outputs = model(**encoded_input)

    # Aggregate the embeddings - here we take the mean across all tokens
    embeddings = outputs.last_hidden_state.mean(dim=1)

    # Convert to one-dimensional array and return
    return embeddings.squeeze().numpy()


def clean_string(text):
    # Replace any character that is not a letter or a space with nothing
    cleaned_text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return cleaned_text

# prepare score for bert similarity
def input_prep_with_embedding(txt,stop_words):

    txt =  clean_string(txt)
    words = word_tokenize(txt)
    # words =[word.strip(u"\u2122").strip(u'\u0256') for word in words]
    filtered_text = [word for word in words if (not word.lower() in stop_words) and (word.isalpha())]
    tagged = pos_tag(filtered_text)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    txt = " ".join(nouns)
    txt = get_embedding(txt)
    txt=reshape_to_2d(txt)

    return txt


def rule_prediction(row, txt):
    score = cosine_similarity(txt,reshape_to_2d(row['training_str_vector']))[0][0]
    return score

# preprocessing text for fuzzywuzzy similarity
def input_prep_fz(txt,stop_words):

    txt =  clean_string(txt)
    words = word_tokenize(txt)
    # words =[word.strip(u"\u2122").strip(u'\u0256') for word in words]
    filtered_text = [word for word in words if (not word.lower() in stop_words) and (word.isalpha())]
    tagged = pos_tag(filtered_text)
    nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    return " ".join(nouns)

# fuzzywuzzy similarity
def rule_prediction_fz(row, txt):
    score = fuzz.partial_ratio(txt.upper(),str(row['training_str']).upper())
    return score

"""
def ensemble_scores(score1, score2):
    # Apply a nonlinear transformation to accentuate extreme values
    transformed_score1 = score1 ** 2
    transformed_score2 = score2 ** 2

    # Compute the average of the transformed scores
    combined_score = (transformed_score1 + transformed_score2) / 2

    # Apply the inverse of the transformation to bring the scale back to [0, 1]
    combined_score = combined_score ** 0.5

    return combined_score
"""

def ensemble_scores(score1, score2):
    # Apply a nonlinear transformation to accentuate extreme values
    transformed_score1 = (score1/100) ** 2
    transformed_score2 = score2 ** 2

    # Compute the average of the transformed scores
    combined_score = (transformed_score1 + transformed_score2) / 2

    # Apply the inverse of the transformation to bring the scale back to [0, 1]
    combined_score = combined_score ** 0.5

    return combined_score


def run_engine(df, input_str, num_results=20):
    
    sw = set(stopwords.words('english'))
    sw.update({"buy", "spend", "select", 'varieties', 'sizes', 'ounce', 'count', 'liter'})
    sw.remove('any')
    
    txt_fz = input_prep_fz(input_str,sw)

    df['pred_score_fz'] = df.apply(lambda row: rule_prediction_fz(row, txt_fz),axis=1)

    txt = input_prep_with_embedding(input_str,sw)

    df['pred_score_bert'] = df.apply(lambda row: rule_prediction(row, txt),axis=1)

    df['combined_score'] = df.apply(lambda row: ensemble_scores(row['pred_score_fz'], row['pred_score_bert']), axis=1)

    results = df.sort_values(by ='combined_score',ascending=False)[['OFFER',"BRAND","RETAILER","combined_score"]].head(num_results)
    
    #print(results[['OFFER',"BRAND","RETAILER", 'combined_score']].to_string(index=False))
    return results[['OFFER',"BRAND","RETAILER", 'combined_score']]