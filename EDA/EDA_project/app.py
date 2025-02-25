import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# Load Titanic dataset
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\ky321\OneDrive\Desktop\Data science and Ai\EDA\EDA_project\titanic dataset.csv")
    return data

data = load_data()

# Title and description
st.title("Exploratory Data Analysis of Titanic Dataset")
st.write("This is an exploratory data analysis of the Titanic dataset.")
st.write("First few rows of the dataset:")
st.dataframe(data.head())

# Data Cleaning Section
st.subheader("Missing Values")
missing_data = data.isnull().sum()
st.write(missing_data)

if st.checkbox('Fill missing Age with median'):
    data['Age'].fillna(data['Age'].median(), inplace=True)

if st.checkbox('Fill missing Embarked with mode'):
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

if st.checkbox('Drop duplicates'):
    data.drop_duplicates(inplace=True)

st.subheader('Cleaned Dataset')
st.dataframe(data.head())

# EDA Section
st.subheader('Statistical Summary of the Data')
st.write(data.describe())

# Age Distribution
st.subheader('Age Distribution')
fig, ax = plt.subplots()
sns.histplot(data['Age'], kde=True, ax=ax)
ax.set_title('Age Distribution')
st.pyplot(fig)

# Gender Distribution
st.subheader('Gender Distribution')
fig, ax = plt.subplots()
sns.countplot(x='Sex', data=data, ax=ax)
ax.set_title('Gender Distribution')
st.pyplot(fig)

# Pclass vs Survived
st.subheader('Pclass vs Survived')
fig, ax = plt.subplots()
sns.countplot(x='Pclass', hue='Survived', data=data, ax=ax)
ax.set_title('Pclass vs Survived')
st.pyplot(fig)

# Survival Count
st.subheader('Survival Count')
fig, ax = plt.subplots()
sns.countplot(x="Survived", data=data, palette="pastel", ax=ax)
ax.set_xticklabels(["Did Not Survive", "Survived"])
ax.set_title("Survival Count on Titanic")
st.pyplot(fig)

# Gender-based Survival Rate
st.subheader('Survival Count by Gender')
fig, ax = plt.subplots()
sns.countplot(x="Sex", hue="Survived", data=data, palette="Set2", ax=ax)
ax.set_title("Survival Count by Gender")
st.pyplot(fig)

# Feature Engineering Section
st.subheader('Feature Engineering: Family Size')
data['FamilySize'] = data['SibSp'] + data['Parch']
fig, ax = plt.subplots()
sns.histplot(data['FamilySize'], kde=True, ax=ax)
ax.set_title('Family Size Distribution')
st.pyplot(fig)

# Correlation Heatmap
st.subheader('Feature Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
ax.set_title("Correlation Heatmap of Titanic Features")
st.pyplot(fig)

# Conclusion Section
st.subheader('Key Insights')
insights = """
- Females have a higher survival rate than males.
- Passengers in 1st class had the highest survival rate.
- The majority of passengers are in Pclass 3.
- Younger passengers tended to survive more often.
- Fare and Class were strong survival indicators, showing that socioeconomic status influenced survival chances.
"""
st.write(insights)
