import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Breast Cancer dataset
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Streamlit app
st.title("Breast Cancer Prediction App (Decision Tree)")

# Add a sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    features = {}
    for i, feature_name in enumerate(data.feature_names):
        min_val = float(data.data[:, i].min())
        max_val = float(data.data[:, i].max())
        mean_val = float(data.data[:, i].mean())
        features[feature_name] = st.sidebar.slider(f'{feature_name} ({min_val:.2f} - {max_val:.2f})', min_val, max_val, mean_val)
    return list(features.values())

user_input = user_input_features()
prediction = clf.predict([user_input])

st.subheader('Class Labels and their corresponding index number')
st.write(data.target_names)

st.subheader('Prediction')
st.write(data.target_names[prediction[0]])

st.subheader('Prediction Probability')
st.write(clf.predict_proba([user_input]))

# Additional Information about the dataset
st.sidebar.subheader('Dataset Information')
st.sidebar.text('Number of classes: 2')
st.sidebar.text('Number of features: {}'.format(len(data.feature_names)))
st.sidebar.text('Number of samples: {}'.format(len(data.data)))
