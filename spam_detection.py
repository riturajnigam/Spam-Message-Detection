# spam_detection.py

# Importing Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset (using sample dataset)
# You can replace this with your own CSV file
url = "https://raw.githubusercontent.com/riturajnigam/sample-datasets/main/spam.csv"
data = pd.read_csv('C:\\Users\\shiva\\Desktop\\webTech\\ML_Projects\\spam.csv', encoding='latin-1')


# Data Preprocessing
data = data[['v1', 'v2']]
data.columns = ['label', 'message']
data['label_num'] = data.label.map({'ham':0, 'spam':1})

# Features and Labels
X = data['message']
y = data['label_num']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Prediction & Accuracy
y_pred = model.predict(X_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Save the model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to predict new message
def predict_message(msg):
    vect_msg = vectorizer.transform([msg])
    prediction = model.predict(vect_msg)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

# Test the function
sample_msg = "Congratulations! You've won a free ticket to Bahamas. Call now!"
print(f"Sample Message Prediction: {predict_message(sample_msg)}")

# Done!
