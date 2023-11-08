
# !pip install torchtext==0.6.0
# !pip install transformers>=4.0.0
# !pip install torch>=1.8.0
# !pip install wordcloud 
# pip install seaborn
#  pip install matplotlib 
# pip install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip install transformers==4.12.0


# #pip install torch==1.8.0+cu110 torchvision==0.9.0+cu110 torchaudio===0.8.0 -f https://download.pytorch.org/whl/cu110/torch_stable.html


# from google.colab import drive
# drive.mount('/content/drive')



import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

import torch
import torchtext
import torchtext.data as data
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, AdamW, DistilBertTokenizer

import seaborn as sns

from textblob import TextBlob
import streamlit as st


# Replace this with the path to your CSV file in Google Drive
file_path = 'reviews_0-250.csv'

# Read the CSV file using pandas
df1 = pd.read_csv(file_path)

#Some additional preprocessing steps
df = df1[:45000]

missing_values = df.isnull().sum()
df = df.dropna()
print("Missing values:\n", missing_values)

duplicates = df.duplicated(subset=['review_text']).sum()
df = df.drop_duplicates()
print("Number of duplicated rows:", duplicates)




# Extract the "review_text" column from the data frame
review_text = df1['review_text'].astype(str)

# Define stopwords to be excluded from the word cloud
stopwords = set(STOPWORDS)
stopwords.update(["br", "href", "don t", "didn t", "don't", "doesn'", "didn't" ])  # Add any other stopwords you'd like to exclude

# Create a word cloud
word_cloud = WordCloud(stopwords=stopwords,
                       max_words=150,
                       background_color='white',
                       colormap='viridis',  # Change the color scheme
                       width=800,
                       height=400).generate(' '.join(review_text))

# # Display the generated word cloud ************************************************************8
# plt.figure(figsize=(12, 6))  # Modify the figure size
# plt.imshow(word_cloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


# Create sentiment labels based on rating
def create_sentiment_label(rating):
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

# Filter relevant columns and create sentiment labels
df = df[['rating', 'review_text']]
df['sentiment'] = df['rating'].apply(create_sentiment_label)

# Save the preprocessed data to new CSV files
df = df[['review_text', 'sentiment']]
df.sample(frac=1).reset_index(drop=True)  # Shuffle the data
train_data, valid_data = train_test_split(df, test_size=0.2, random_state=42)

train_data.to_csv("train.csv", index=False)
valid_data.to_csv("valid.csv", index=False)



# class_counts = df['sentiment'].value_counts()
# print("Class distribution:\n", class_counts)

# plt.figure(figsize=(10, 5))
# sns.barplot(x=class_counts.index, y=class_counts.values)
# plt.title("Sentiment Class Distribution")
# plt.xlabel("Sentiment")
# plt.ylabel("Number of Samples")
# plt.show()


# Read the CSV file
#df = pd.read_csv("C:/Users/ammar/Downloads/archive/reviews_0_250.csv")

#limiting the dataset for faster training - >>>>> REMOVE THIS <<<<<
#df = df1[:500]

#df = df1[:5000]

# Tokenizer setup
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

max_input_length = tokenizer.max_model_input_sizes['distilbert-base-multilingual-cased']

def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens

# Define torchtext Fields
TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=tokenize_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  init_token=init_token_idx,
                  eos_token=eos_token_idx,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx)

LABEL = data.LabelField(dtype=torch.float)

# Create Dataset
fields = [('text', TEXT), ('label', LABEL)]
train_data, valid_data = data.TabularDataset.splits(
    path='',
    train='train.csv',
    validation='valid.csv',
    format='csv',
    fields=fields,
    skip_header=True
)

# Build the vocabulary for the LABEL field
LABEL.build_vocab(train_data)

# Set up DataLoader
BATCH_SIZE = 16

if torch.cuda.is_available():
    print("GPU is available. Device:", torch.cuda.get_device_name(0))
    device = torch.device('cuda:0')
else:
    print("Using CPU.")
    device = torch.device('cpu')


train_iterator, valid_iterator = torchtext.data.BucketIterator.splits(
    (train_data, valid_data),
    batch_size=BATCH_SIZE,
    device=device,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True
)

# Load the model and set it up for training
model = DistilBertForSequenceClassification.from_pretrained(
    'philschmid/distilbert-base-multilingual-cased-sentiment-2',
    num_labels=len(LABEL.vocab)
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=1e-7, weight_decay=3e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training and validation functions
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text)[0]
        loss = criterion(predictions, batch.label.long())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)[0]
            loss = criterion(predictions, batch.label.long())
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Train the model
# NUM_EPOCHS = 25

# for epoch in range(NUM_EPOCHS):
#     train_loss = train(model, train_iterator, optimizer, criterion)
#     valid_loss = evaluate(model, valid_iterator, criterion)
#     print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}')

       
#SIMPLIFIED PREDITOR
def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.argmax(model(tensor)[0], dim=1)
    return prediction.item()

# Example usage
example_review = "This lip balm is the best I ever had. It works. It hydrates and last long. It has everything you need in a balm. It even has a little spatula that you can use to scoop the product easily and cleanly."
sentiment = predict_sentiment(model, tokenizer, example_review)

sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
print(f'Sentiment: {sentiment_mapping[sentiment]}')

#RAW SCORE PREDICTOR (Probability) FOR EACH CLASS
def predict_sentiment2(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    logits = model(tensor)[0]
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1)
    return prediction.item(), probabilities.squeeze().tolist()

# Example usage
example_review = "I dont like this product!"
sentiment, sentiment_probabilities = predict_sentiment2(model, tokenizer, example_review)

sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
print(f'Sentiment: {sentiment_mapping[sentiment]}')
print(f'Probabilities: {sentiment_probabilities}')

sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
rev1 = "Im unsure about my thoughts regarding this lipstick"
rev2 = "This cream has changed my life!"
rev3 = "Absolute garbage eyeliner"
rev4 = "Very bad experience overall with this skincare company"
rev5 = "I dont know if i would recommend this toner to someone, its fine"
rev6 = "This product isnt bad but not good either"
rev7 = "I would purchase it again"


sentiment1 = predict_sentiment(model, tokenizer, rev1)

sentiment2 = predict_sentiment(model, tokenizer, rev2)
sentiment3 = predict_sentiment(model, tokenizer, rev3)
sentiment4 = predict_sentiment(model, tokenizer, rev4)
sentiment5 = predict_sentiment(model, tokenizer, rev5)
sentiment6 = predict_sentiment(model, tokenizer, rev6)
sentiment7 = predict_sentiment(model, tokenizer, rev7)


print(f'Sentiment1: {sentiment_mapping[sentiment1]}') #Expected: Neutral
print(f'Sentiment2: {sentiment_mapping[sentiment2]}') #Expected: Positive
print(f'Sentiment3: {sentiment_mapping[sentiment3]}') #Expected: Negative
print(f'Sentiment4: {sentiment_mapping[sentiment4]}') #Expected: Negative
print(f'Sentiment5: {sentiment_mapping[sentiment5]}') #Expected: Neutral
print(f'Sentiment6: {sentiment_mapping[sentiment6]}') #Expected: Neutral
print(f'Sentiment7: {sentiment_mapping[sentiment7]}') #Expected: Positive

st.title("Sentiment Analysis")
# label=st.subheader("Feedback")

area=st.text_area(" ", value="", height=200, max_chars=None, key=None, help=None,placeholder="Hi Welcome Please Enter the Sentence....")


if st.button("Predict Sentiment"):
    if len(area) > 20:
        sentiment = predict_sentiment(model, tokenizer, area)
        sentiment_label = sentiment_mapping[sentiment]
        st.subheader(f"Predicted Sentiment: {sentiment_label}")
    else:
        st.subheader("Enter at least 20 characters for prediction.")


