import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')

# Print the column names and the first few rows to confirm the structure
print("Column Names:")
print(data.columns)
print("\nFirst Few Rows:")
print(data.head())

# Keep only the necessary columns and drop any unnecessary ones
data = data[['label', 'text', 'label_num']]  # Adjust based on actual column names

# Rename columns if necessary
data.columns = ['label', 'message', 'label_num']

# Clean the text data
def clean_text(text):
    return text.lower()

data['message'] = data['message'].apply(clean_text)

# Split data into features (X) and labels (y)
X = data['message']
y = data['label_num']

# Convert text data to numerical format using CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model and print accuracy
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')

# Save the model and the vectorizer for use in the Flask app
with open('spam_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('cv.pkl', 'wb') as cv_file:
    pickle.dump(cv, cv_file)
