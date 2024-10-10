import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load and Preprocess the Dataset
data = pd.read_csv('dataset.csv')

# Drop the 'Unnamed: 0' column if it's not needed
data = data.drop(columns=['Unnamed: 0'])

# Rename 'label' to 'v1' and 'text' to 'v2'
data = data.rename(columns={"label": "v1", "text": "v2"})

# Step 2: Prepare Features and Labels
X = data['v2']  # Features (message text)
y = data['v1'].map({'ham': 0, 'spam': 1})  # Labels (map 'ham' to 0 and 'spam' to 1)

# Step 3: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Vectorize the Text Data
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train the Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Evaluate the Model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the Model and Vectorizer
joblib.dump(model, 'spam_detector_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
