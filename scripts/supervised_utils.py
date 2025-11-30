import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional

tweet_tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()

def load_data(data_path='../data/twitter_training.csv', text_column='Tweet'):
    """Loads the dataset and prepares the DataFrame."""
    try:
        df = pd.read_csv(data_path, header=None, encoding='latin1')
        df.columns = ['ID', 'Entity', 'Sentiment', text_column]
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}.")
        return None

def get_wordnet_pos(tag):
    """Map NLTK POS tags to WordNet POS tags"""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def extract_features(text):
    hashtags = re.findall(r'#(\w+)', text)
    mentions = re.findall(r'@(\w+)', text)
    return ' '.join(hashtags), ' '.join(mentions)

def clean_and_tokenize(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    tokens = tweet_tokenizer.tokenize(text)
    return tokens

def lemmatize_tokens(tokens):
    """Lemmatizes tokens using POS tagging for context-aware normalization."""
    tagged_tokens = nltk.pos_tag(tokens)
    lemmatized_tokens = []
    for word, tag in tagged_tokens:
        wntag = get_wordnet_pos(tag)
        if isinstance(word, str):
            lemma = lemmatizer.lemmatize(word, pos=wntag)
            lemmatized_tokens.append(lemma)
    return lemmatized_tokens

def preprocess_data(df, text_column='Tweet'):
    df[text_column] = df[text_column].fillna('')
    df['hashtags'], df['mentions'] = zip(*df[text_column].apply(extract_features))
    df['tokens'] = df[text_column].apply(clean_and_tokenize)
    df['lemmas'] = df['tokens'].apply(lemmatize_tokens)
    df['processed_text'] = df['lemmas'].apply(lambda x: ' '.join(x))
    return df

def vectorize_data(df, text_column='processed_text'):
    tfidf_vectorizer = TfidfVectorizer(sublinear_tf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_df, tfidf_vectorizer

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('#', ' ')
    text = re.sub(r"[^a-z0-9\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_df(df: pd.DataFrame, text_column: str = 'Tweet') -> pd.DataFrame:
    df = df.copy()
    df[text_column] = df[text_column].fillna('')
    df['processed_text'] = df[text_column].apply(preprocess_text)
    df['Sentiment'] = df['Sentiment'].replace({'Irrelevant': 'Neutral'})
    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['label'] = df['Sentiment'].map(sentiment_map).fillna(-1).astype(int)
    df = df.dropna(subset=[text_column]).reset_index(drop=True)
    return df
