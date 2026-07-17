import pandas as pd
import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("drugsComTrain_raw.csv") #loads raw data to data frame
df = df.dropna(subset=["review"]) #drops NA valued rows from the review column

print("Loaded raw data:", df.shape) #prints number of rows and columns in the data frame

def clean_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii','ignore').decode('utf-8','ignore') #everything is converted to plain ASCII
    text = re.sub(r'[^a-zA-Z\s]', '', text) #removes other characters
    text = text.lower()  #to lowercase
    tokens = nltk.word_tokenize(text) #tokenization
    tokens = [word for word in tokens if word not in stop_words] #remove stop words 
    tokens = [lemmatizer.lemmatize(word) for word in tokens] #lemmatize to base word
    return " ".join(tokens)

print("Cleaning review text... this may take a few minutes")
df["cleaned_review"] = df["review"].apply(clean_text)

print("\n--- Sample before/after ---")
for i in range(3):
    print(f"\nOriginal:  {df['review'].iloc[i][:150]}")
    print(f"Cleaned:   {df['cleaned_review'].iloc[i][:150]}")

df.to_csv("drugsComTrain_cleaned.csv", index=False)
print(f"\nSaved cleaned data to drugsComTrain_cleaned.csv ({df.shape[0]} rows)")
