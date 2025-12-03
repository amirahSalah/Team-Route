import streamlit as st
import pandas as pd
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


nltk_packages = ['punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4', 'stopwords']
for pkg in nltk_packages:
    try:
        nltk.data.find(pkg)
    except Exception:
        nltk.download(pkg)

tokenizer = TweetTokenizer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def get_wordnet_pos(tag):
    """Map POS tag to wordnet tag for lemmatization."""
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    return wordnet.NOUN

def clean_text(text: str) -> str:
    """Basic cleaning: lower, remove URLs, remove mentions (but keep mention text in features), remove punctuation (keep hashtag word)."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ')
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)      
   
    text = re.sub(r'#', '', text)
 
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def extract_hashtags(text: str):
    return re.findall(r'#(\w+)', text)

def extract_mentions(text: str):
    return re.findall(r'@(\w+)', text)

def tokenize(text: str):
    if not isinstance(text, str):
        return []
    return tokenizer.tokenize(text)

def lemmatize_tokens(tokens):
    if not tokens:
        return []
    tagged = nltk.pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(tag)) for w, tag in tagged]
   
    lemmas = [w for w in lemmas if w not in stop_words]
    return lemmas

def preprocess_row(text):
    hashtags = extract_hashtags(text)
    mentions = extract_mentions(text)
    cleaned = clean_text(text)
    toks = tokenize(cleaned)
    lemmas = lemmatize_tokens(toks)
    processed_text = ' '.join(lemmas)
    return {
        'text': text,
        'cleaned': cleaned,
        'hashtags': ','.join(hashtags),
        'mentions': ','.join(mentions),
        'tokens': toks,
        'lemmas': lemmas,
        'processed_text': processed_text
    }


st.set_page_config(page_title="Preprocessing - Demo", layout="wide")
st.title("Text Preprocessing (Tweet-focused)")

st.markdown("This app performs cleaning, tokenization, lemmatization, and stopword removal. Upload a CSV with a `Tweet` column for batch processing or type a single tweet below.")


with st.sidebar:
    st.header("Options")
    show_tfidf = st.checkbox("Compute TF-IDF (sample)", value=False)
    sublinear_tf = st.checkbox("TF-IDF sublinear_tf=True", value=True)
    tfidf_max_features = st.number_input("TF-IDF max features (for sample)", min_value=100, max_value=20000, value=2000, step=100)

left, right = st.columns([1, 1])
with left:
    st.subheader("Process single text")
    input_text = st.text_area("Enter text / tweet here:", height=140)
    if st.button("Process text"):
        if not input_text or input_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            res = preprocess_row(input_text)
            st.subheader("Results")
            st.markdown("**Original text:**")
            st.write(res['text'])
            st.markdown("**Cleaned text:**")
            st.write(res['cleaned'])
            st.markdown("**Hashtags:**")
            st.write(res['hashtags'] or "—")
            st.markdown("**Mentions:**")
            st.write(res['mentions'] or "—")
            st.markdown("**Tokens:**")
            st.write(res['tokens'])
            st.markdown("**Lemmas (stopwords removed):**")
            st.write(res['lemmas'])
            st.markdown("**Processed text:**")
            st.success(res['processed_text'])

with right:
    st.subheader("Batch processing (CSV)")
    st.info("CSV must contain a column named 'Tweet' (case-sensitive).")
    uploaded = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            df = None

        if df is not None:
            if 'Tweet' not in df.columns:
                st.error("CSV does not contain a 'Tweet' column.")
            else:
                n_preview = st.number_input("Preview rows", min_value=1, max_value=500, value=5)
                st.write("Preview of original column:")
                st.write(df['Tweet'].head(n_preview))

                if st.button("Run preprocessing on uploaded CSV"):
                    processed = []
                    for t in df['Tweet'].astype(str).tolist():
                        processed.append(preprocess_row(t))
                    proc_df = pd.DataFrame(processed)
                   
                    result_df = pd.concat([df.reset_index(drop=True), proc_df.reset_index(drop=True)], axis=1)
                    st.success(f"Processed {len(result_df)} rows.")
                    st.write(result_df.head(n_preview))

                
                    csv_bytes = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download processed CSV", csv_bytes, "processed_tweets.csv", "text/csv")

                    if show_tfidf:
                        st.markdown("#### TF-IDF sample (first rows × first 10 features)")
                        corpus = result_df['processed_text'].astype(str).tolist()
                        vec = TfidfVectorizer(max_features=int(tfidf_max_features), sublinear_tf=sublinear_tf, stop_words='english')
                        X = vec.fit_transform(corpus)
                        features = vec.get_feature_names_out()[:10]
                        tfidf_df = pd.DataFrame(X.toarray()[:min(5, X.shape[0]), :10], columns=features)
                        st.write(tfidf_df)


st.markdown("---")
st.markdown("**Notes:** This tool performs basic preprocessing for prototyping, including stopword removal. For production, extend tokenization, handle emojis, normalise slang, and move heavy processing to background workers.")


