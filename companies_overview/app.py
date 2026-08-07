import pandas as pd
import streamlit as st

st.title("Company Reviews Analyzer")
data = pd.read_csv("companies_overview/companies.csv")
st.write(data.head())
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Ratings Distribution")
fig, ax = plt.subplots()
sns.countplot(x="Rating", data=data, ax=ax, palette="viridis")
st.pyplot(fig)

st.subheader("Review Length Distribution")
data['Reviews'] = data['Reviews'].apply(lambda x: len(str(x).split()))
fig, ax = plt.subplots()
ax.hist(data['Reviews'], bins=30, color="skyblue", edgecolor="black")
st.pyplot(fig)
data['Reviews'] = data['Reviews'].apply(lambda x: len(str(x).split()))
X = data[['Reviews']]   # simple numeric feature
y = (data['Rating'] > 3).astype(int)  # sentiment: 1=positive, 0=negative
from sklearn.model_selection import train_test_split
# Create binary sentiment labels
y = data['Rating'].apply(lambda r: 1 if r >= 4 else 0)
# Check distribution
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

st.text("Logistic Regression Report:\n" + classification_report(y_test, y_pred_log))
st.text("Naive Bayes Report:\n" + classification_report(y_test, y_pred_nb))
st.text("KNN Report:\n" + classification_report(y_test, y_pred_knn))
st.subheader("Try Your Own Review")
user_input = st.text_area("Enter a company review:")

if user_input:
    length = len(user_input.split())
    prediction = log_reg.predict([[length]])[0]
    st.write("Predicted Sentiment:", "Positive" if prediction == 1 else "Negative")


