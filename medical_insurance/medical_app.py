import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score

# Load dataset (replace with your own CSV if needed)
@st.cache_data
def load_data():
    return pd.read_csv("medical_insurance/medical-charges.csv")

df = load_data()
st.title("ML Models Frontend")
st.write("Choose a model and run predictions/evaluations.")

# Sidebar for model selection
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression (charges prediction)",
     "Logistic Regression (smoker classification)",
     "KNN (classification)",
     "Naive Bayes (classification)",
     "Algorithm Comparison"]
)

# Preprocessing
df_proc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_proc.drop("charges", axis=1))
y_reg = df_proc["charges"]
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)

# For classification (smoker column as target)
df_class = pd.get_dummies(df, columns=["sex", "region"], drop_first=True)
df_class["smoker"] = df_class["smoker"].map({"yes":1, "no":0})
X_class = df_class.drop("smoker", axis=1)
y_class = df_class["smoker"]
X_train, X_test, y_train, y_test = train_test_split(scaler.fit_transform(X_class), y_class, test_size=0.2, random_state=42)

# Model logic
if model_choice == "Linear Regression (charges prediction)":
    model = LinearRegression()
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)
    st.write("### Linear Regression Results")
    st.write("Intercept:", model.intercept_)
    st.write("MSE:", mean_squared_error(y_test_reg, y_pred))
    st.write("R² Score:", r2_score(y_test_reg, y_pred))

elif model_choice == "Logistic Regression (smoker classification)":
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Logistic Regression Results")
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

elif model_choice == "KNN (classification)":
    k = st.sidebar.slider("Select k", 3, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write(f"### KNN Results (k={k})")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))

elif model_choice == "Naive Bayes (classification)":
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.write("### Naive Bayes Results")
    st.write("Confusion Matrix:", confusion_matrix(y_test, y_pred))
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred))
    st.write("Recall:", recall_score(y_test, y_pred))
    st.write("F1 Score:", f1_score(y_test, y_pred))

elif model_choice == "Algorithm Comparison":
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results.append([name,
                        accuracy_score(y_test, y_pred),
                        precision_score(y_test, y_pred),
                        recall_score(y_test, y_pred),
                        f1_score(y_test, y_pred)])
    df_results = pd.DataFrame(results, columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1 Score"])
    st.write("### Algorithm Comparison")
    st.dataframe(df_results)
