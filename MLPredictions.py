import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from fuzzywuzzy import fuzz

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming the CSV file is located in the same directory as the script
file_path = os.path.join(script_dir, 'final_dataset.csv')

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Extract features (X) and labels (y)
X = df['CodeSnippet']  # Exclude the last two columns (code snippets)
y = df['label']   # Labels: Last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#
# Vectorize the code snippets using TF-IDF
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(X_train_vectorized[0])
# Train the KNN classifier
k = 3  # Choose the number of neighbors (you can experiment with different values)
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train_vectorized, y_train)

# Predict labels for the test set
y_pred = knn_classifier.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn_classifier.classes_, yticklabels=knn_classifier.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Preprocess data if necessary

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Define features and labels
X_train_text = train_data['CodeSnippet']  # Text column
X_train_label = train_data['label']  # Label column (from 0 to 6)
X_test_text = test_data['CodeSnippet']  # Text column
X_test_label = test_data['label']  # Label column (from 0 to 6)

# Define a function to calculate fuzzy partial ratio
def fuzzy_partial_ratio(text, label):
    return fuzz.partial_ratio(str(text), str(label))

# Calculate fuzzy partial ratio for training data
X_train_partial_ratio = [fuzzy_partial_ratio(text, label) for text, label in zip(X_train_text, X_train_label)]

# Calculate fuzzy partial ratio for testing data
X_test_partial_ratio = [fuzzy_partial_ratio(text, label) for text, label in zip(X_test_text, X_test_label)]

# Create DataFrame with features including fuzzy partial ratio
X_train_features = pd.DataFrame({
    'fuzzy_partial_ratio': X_train_partial_ratio,
    # Add other features if available
})

X_test_features = pd.DataFrame({
    'fuzzy_partial_ratio': X_test_partial_ratio,
    # Add other features if available
})

# Define labels
y_train = train_data['label']
y_test = test_data['label']

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_features, y_train)

# Evaluate the model
accuracy = model.score(X_test_features, y_test)
print("Fuzzy Accuracy:", accuracy)

# Random Forest
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train_vectorized, y_train)
rf_accuracy = rf_classifier.score(X_test_vectorized, y_test)
print("Random Forest Accuracy:", rf_accuracy)

# SVM
svm_classifier = SVC()
svm_classifier.fit(X_train_vectorized, y_train)
svm_accuracy = svm_classifier.score(X_test_vectorized, y_test)
print("SVM Accuracy:", svm_accuracy)

# Naive Bayes
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)
nb_accuracy = nb_classifier.score(X_test_vectorized, y_test)
print("Naive Bayes Accuracy:", nb_accuracy)

# Gradient Boosting
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_vectorized, y_train)
gb_accuracy = gb_classifier.score(X_test_vectorized, y_test)
print("Gradient Boosting Accuracy:", gb_accuracy)

# Neural Network (MLP)
mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train_vectorized, y_train)
mlp_accuracy = mlp_classifier.score(X_test_vectorized, y_test)
print("Neural Network Accuracy:", mlp_accuracy)

# Extract features (X) and labels (y)
X = df['CodeSnippet']  # Features: Code snippets
y = df['label']         # Labels: Last column

# Tokenize and pad sequences
max_features = 10000  # Number of words to consider as features
max_len = 500         # Max sequence length

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=max_len)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=128))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(7, activation='softmax'))  # Adjust output dimension for your number of classes

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

