import pandas as pd

df= pd.read_csv("ai_student_impact_dataset.csv")

print(df.describe())

# check Perceived_AI_Dependency,Institutional_Policy,Anxiety_Level_During_Exams

print(df["Perceived_AI_Dependency"].value_counts())
print(df["Institutional_Policy"].value_counts())    
print(df["Anxiety_Level_During_Exams"].value_counts())  

print(df.corr(numeric_only=True))