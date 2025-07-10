import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import kagglehub
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
#!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/
from sklearn.model_selection import ParameterGrid
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Embedding, Flatten, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix

english_stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def show_class_distribution(df):
    '''Show the distribution of classes in a dataframe.'''
    sns.countplot(x=df['generated'])
    plt.title('Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def sample_dataframe(df, total_samples=300000):
    '''Sample the dataframe to have an equal number of each class.'''
    df_zero = df[df['generated'] == 0]
    df_one = df[df['generated'] == 1]

    df_zero_sampled = df_zero.sample(int(total_samples/2), random_state=1)
    df_one_sampled = df_one.sample(int(total_samples/2), random_state=1)

    df = pd.concat([df_zero_sampled, df_one_sampled])
    df.reset_index(inplace=True)
    print(f'Number of rows in data subset: {len(df)}')
    return df

def get_first_sentence(text):
    '''Get the first sentence of a text.'''
    parts = re.split(r'(\. |\? |! )', text, maxsplit=1)
    if len(parts) > 1:
        return parts[0]
    return ""  # No split found

def clean_text(text):
    '''Clean text by removing non-alphanumeric characters, extra spaces, and stopwords.'''
    if not isinstance(text, str):  # Make sure text is a string
        return ""
    
    text = text.lower() # Convert text to lowercase
    text = re.sub(r'\W', ' ', text) # Remove non-alphanumeric characters  
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in english_stopwords]) # Remove stop words
    return text

def create_model(embedding_size=128, filters=64, dropout_rate=0.5):
    '''Create a 1D Convolutional Neural Network model.'''
    model = Sequential([
        Embedding(vocab_size, embedding_size),
        Conv1D(filters, 3, activation="relu"),
        MaxPooling1D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(dropout_rate),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

def grid_search(model_function, param_grid, X_train, y_train, X_val, y_val, epochs=5, batch_size=32):
    '''Perform grid search over hyperparameters for a model.'''

    params = list(ParameterGrid(param_grid))
    
    best_score = -np.inf
    best_params = None
    best_model = None
    
    for params in params:
        print(f"Training with parameters: {params}")
        
        # Create the model with the current parameters
        model = model_function(**params)
        
        # Train the model
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1)
        
        # Evaluate the model on the validation data
        val_preds = model.predict(X_val)
        val_preds = (val_preds > 0.5).astype("int32")
        val_score = accuracy_score(y_val, val_preds)
        
        print(f"Validation Accuracy: {val_score:.4f}")
        
        # Check if this model has the best validation score
        if val_score > best_score:
            best_score = val_score
            best_params = params
            best_model = model
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Validation Accuracy: {best_score:.4f}")
    
    # Return the best model and its parameters
    return best_model, best_params, best_score

def save_best_results(best_params,
                      best_score,
                      results_dir="results"):
    '''Save the best hyperparameters, accuracy, and test results to a directory.'''

    # Create results directory if It doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Save best parameters and accuracy
    best_params = {
        "embedding_size": best_params["embedding_size"],
        "filters": best_params["filters"],
        "dropout_rate": best_params["dropout_rate"],
        "best_val_accuracy": best_score
    }

    with open(os.path.join(results_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Train final model with best hyperparameters
    final_model = create_model(embedding_size=best_params["embedding_size"],
                            filters=best_params["filters"],
                            dropout_rate=best_params["dropout_rate"])

    history = final_model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=5, batch_size=32)

    # Evaluate on Test Data
    test_loss, test_acc = final_model.evaluate(X_test_pad, y_test)
    test_results = {"test_loss": test_loss, "test_accuracy": test_acc}

    with open(os.path.join(results_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=4)

    # Predict on Test Data
    y_pred_prob = final_model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype("int32")

    # Save Classification Report
    classification_rep = classification_report(y_test, y_pred, target_names=["Human", "AI"])
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(classification_rep)

    # Save Confusion Matrix as an Image
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))  # Save image
    plt.close()

    # Save Training History (Loss & Accuracy Plots)
    plt.figure(figsize=(10, 4))

    # Plot Training & Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Training & Validation Accuracy")

    # Plot Training & Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training & Validation Loss")

    # Save Plot
    plt.savefig(os.path.join(results_dir, "training_history.png"))
    plt.close()


# Load Data
path = f"{os.getcwd()}\\ProjRestanta\\AI_Human.csv"
# path = kagglehub.dataset_download("shanegerami/ai-vs-human-text") + "/AI_Human.csv"
df = pd.read_csv(path)

# Show class distribution
show_class_distribution(df)

df = sample_dataframe(df, total_samples=300000)

# Preprocess text data
tqdm.pandas()
df["text"] = df["text"].progress_apply(get_first_sentence)
df["text"] = df["text"].progress_apply(clean_text)

# Split data into train, validation, and test sets
texts = df["text"].astype(str)
labels = df["generated"]
X_train, X_temp, y_train, y_temp = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# Tokenization (no pretrained embeddings)
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for OOV token
print(f'Vocabulary size: {vocab_size}')

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(df['text'])
max_length = max(len(seq) for seq in sequences)  # Find max sequence length
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")
print(f'Max sequence length: {max_length}')

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding="post")
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding="post")

# Convert labels to numpy array
y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

model = create_model(embedding_size=128, filters=64, dropout_rate=0.5)
# Train Model
history = model.fit(
    X_train_pad, y_train,
    validation_data=(X_val_pad, y_val),
    epochs=5, batch_size=32
)

y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype("int32")

# Evaluate Model
classification_rep = classification_report(y_test, y_pred, target_names=["Human", "AI"])


# Perform grid search
param_grid = {
    "embedding_size": [32, 64, 128],
    "filters": [32, 64, 128],
    "dropout_rate": [0.3, 0.5, 0.7]
}

best_model, best_params, best_score = grid_search(create_model, param_grid, X_train_pad, y_train, X_val_pad, y_val, epochs=5, batch_size=32)

save_best_results(best_params, best_score, results_dir="results")