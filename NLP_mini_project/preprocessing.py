import contractions
from unidecode import unidecode
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Remove Newlines 
def remove_lines(data):
    clean_text = data.replace("\n"," ").replace("\\n" , " ").replace("\t"," ")
    return clean_text

# Contraction mapping 
def expand_text(data):
    expanded_doc = contractions.fix(data)
    return expanded_doc

# Handle accented characters
def accented_char(data):
    fixed_text = unidecode(data)
    return fixed_text

# Clean data 
stopword_list = stopwords.words("english")
stopword_list.remove("no")
stopword_list.remove("nor")
stopword_list.remove("not")

def clean_data(data):
    tokens = word_tokenize(data)
    clean_text = [word.lower() for word in tokens if (word not in punctuation) and (word.lower() not in stopword_list) and (len(word)>2) and (word.isalpha())]
    return clean_text

# Lemmatization 
def lemmatization(data):
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for word in data:
        lemmatized_word = lemmatizer.lemmatize(word)
        final_text.append(lemmatized_word)

    return final_text

def join_list(data):
    return " ".join(data)
