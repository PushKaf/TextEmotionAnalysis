from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from typing import Union, List
import pandas as pd
import pickle, re, nltk


nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

class Model:
    def __init__(self) -> None:
        # Constansts and needed variables
        self.MODELS_FOLDER = "models"
        self.EMOTIONS_MAP = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        self.twitter_pattern = re.compile(r'@[\w]+')
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer= WordNetLemmatizer()

        # Grab the data file
        reviews = pd.read_csv('finalData.csv')
        reviews.dropna(inplace=True)

        # Basically do everything we did for training (this will be used to process user given data)
        self.X = reviews['text']  # Features (preprocessed text)
        self.y = reviews['label']  # Target labels

        self.X_train, _, _, _ = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.count_vectorizer = CountVectorizer(max_features=10000)  # Adjust max_features as needed
        self.count_vectorizer.fit_transform(self.X_train)

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.X_train)

    def load_ml_model(self, pickle_filename):
        # Given the pickle file, it will load and return the model

        with open(f"{self.MODELS_FOLDER}/{pickle_filename}.pkl", "rb") as fin:
            model = pickle.load(fin)

        return model

    def load_dl_model(self, h5_file):
        # Given h5 file, loads and returns model

        return load_model(f"{self.MODELS_FOLDER}/{h5_file}.h5")

    def _mapNumToEmotion(self, num) -> str:
        # Maps the number to the emotion

        return self.EMOTIONS_MAP[num]

    def predict(self, model, prediction_text, is_ml=True, multiple=False, max_seq_len=250) -> Union[str, List[dict]]:
        # Given the model, text to predict, and if it's ML or DL, it will predict and return
        # JSON format file with information about your query

        if is_ml:
            return self._predict_ml(model, prediction_text, multiple=multiple)

        return self._predict_dl(model, prediction_text, max_seq_len=max_seq_len)

    def _predict_ml(self, model, prediction_text, multiple=False) -> Union[str, List[dict]]:
        # Preprocess and vectorize text, then return JSON format if multiple or just emotion if not

        preprocess_user_text = self._preprocess_text(prediction_text)
        vectorized_text = self.count_vectorizer.transform([preprocess_user_text])

        if not multiple:
            return self._mapNumToEmotion(model.predict(vectorized_text)[0])
        else:
            res = model.predict_proba(vectorized_text)[0]
            return self._format_predict_multiple(res)

    def _predict_dl(self, model, prediction_text, max_seq_len = 250) -> List[dict]:
        # Process and pad sequences then predict and return the JSON object

        preprocessed = self._preprocess_text(prediction_text)
        seq = self.tokenizer.texts_to_sequences([preprocessed])

        processed_text = pad_sequences(seq, maxlen=max_seq_len)

        res = model.predict(processed_text)[0]
        return self._format_predict_multiple(res)

    def _preprocess_text(self, text) -> str:
        text = text.lower() # convert to lowercase
        text = self.twitter_pattern.sub("", text) # remove twitter handles
        text = self.url_pattern.sub("", text) # remove urls
        text = re.sub(r'[^a-zA-Z\s]', '', text) # remove special chars

        # Tokenization
        tokens = word_tokenize(text)
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words] # remove stopwords and lemmatize

        # Join tokens back into a string
        processed_text = ' '.join(filtered_tokens)

        return processed_text

    def _format_predict_multiple(self, model_res) -> List[dict]:
        result = []

        for idx, val in enumerate(model_res):
            data = {"emotion": self._mapNumToEmotion(idx), "confidence": float(f"{val*100:.3f}")}

            result.append(data)

        return sorted(result, key=lambda x: x["confidence"], reverse=True)

    def accuracyReport(self, y_pred) -> None:
        accuracy_logistic_regression = accuracy_score(self.y_test, y_pred)

        print("Accuracy:", accuracy_logistic_regression)
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))