def get_year_of_study_map():
    name_year={
        "Junior": 3 ,
        "Freshman" : 1,
        "Senior"  : 4, 
        "Sophomore"  : 2,
        "Graduate" : 5,
    }
    return name_year
def get_prompt_level_map():
    level_to_int={
    "Beginner": 1 ,
    "Intermediate" : 2,
    "Advanced"  : 3, 
    }
    return level_to_int

def get_Study_to_AI_Ratio(Traditional_Study_Hours,Weekly_GenAI_Hours ):
    return Traditional_Study_Hours/ (Weekly_GenAI_Hours+1)

def get_Stress_Index(Weekly_GenAI_Hours,Anxiety_Level_During_Exams,Pre_Semester_GPA, Paid_Subscription):
    gpa_inverso = 4.0 - Pre_Semester_GPA
    moltiplicatore_premium = 2.5 if Paid_Subscription == 1 else 1.0
    prodotto = Weekly_GenAI_Hours * Anxiety_Level_During_Exams * gpa_inverso * moltiplicatore_premium
    return prodotto

def get_Anxiety_by_Year(Anxiety_Level_During_Exams,Year_of_Study):
    return Anxiety_Level_During_Exams * Year_of_Study

def get_Study_Imbalance(Traditional_Study_Hours, Weekly_GenAI_Hours):
    return Traditional_Study_Hours - Weekly_GenAI_Hours


def get_assegna_fascia_gpa(gpa):
    if gpa <= 1.5:
        return 0  # Mediocre / Basso GPA
    elif gpa <= 3:
        return 1  # Buono / Medio GPA
    else:
        return 2  # Eccellente / Alto GPA

def get_Study_Imbalance(Traditional_Study_Hours,Weekly_GenAI_Hours):
    return Traditional_Study_Hours - Weekly_GenAI_Hours
