def process_user_data_for_gpa_model(major, year, pre_gpa, policy, study_hours, weekly_ai_hours, use_case, paid_sub, anxiety, ai_dep, prompt_skill, tool_div):
    features_comuni = {
    'Year_of_Study': year,
    'Pre_Semester_GPA': pre_gpa,
    'Weekly_GenAI_Hours': weekly_ai_hours,
    'Prompt_Engineering_Skill': prompt_skill,
    'Tool_Diversity': tool_div,
    'Paid_Subscription': paid_sub,
    'Traditional_Study_Hours': study_hours,
    'Perceived_AI_Dependency': ai_dep,
    'Anxiety_Level_During_Exams': anxiety,
    # booleani caso d'uso inizializzati a false
    'is_Copywriting/Drafting': False,
    'is_Ideation': False,
    'is_Summarizing_Reading': False,
    'is_Debugging/Troubleshooting': False,
    'is_Direct_Answer_Generation': False,
    # inizializzazione policy istituzionali
    'is_Allowed_With_Citation': False,
    'is_Strict_Ban': False,
    'is_Actively_Encouraged': False,
    # inizializzazione major bool
    'major_is_Humanities': False,
    'major_is_Medical': False,
    'major_is_Business': False,
    'major_is_STEM': False,
    'major_is_Arts': False
}

# Attiviamo il flag corretto in base all'input selezionato dall'utente

    match use_case:
        case "Copywriting/Drafting":
            features_comuni['is_Copywriting/Drafting'] = True
        case "Ideation":
            features_comuni['is_Ideation'] = True
        case "Summarizing_Reading":
            features_comuni['is_Summarizing_Reading'] = True
        case "Coding":
            features_comuni['is_Debugging/Troubleshooting'] = True
        case "Research":
            features_comuni['is_Direct_Answer_Generation'] = True

    match policy:
        case "Allowed_With_Citation":
            features_comuni['is_Allowed_With_Citation'] = True
        case "Strict_Ban":
            features_comuni['is_Strict_Ban'] = True
        case "Actively_Encouraged":
            features_comuni['is_Actively_Encouraged'] = True

    match major:
        case "Humanities":
            features_comuni['major_is_Humanities'] = True
        case "Medical":
            features_comuni['major_is_Medical'] = True
        case "Business":
            features_comuni['major_is_Business'] = True
        case "STEM":
            features_comuni['major_is_STEM'] = True
        case "Arts":
            features_comuni['major_is_Arts'] = True
    return features_comuni



def process_user_data_for_burnout_model(major, year, pre_gpa, policy, study_hours, weekly_ai_hours, use_case, paid_sub, anxiety, ai_dep, prompt_skill, tool_div):
    features_comuni = {
    'Post_Semester_GPA':None,
    'Year_of_Study': year,
    'Pre_Semester_GPA': pre_gpa,
    'Weekly_GenAI_Hours': weekly_ai_hours,
    
    'Tool_Diversity': tool_div,
    'Paid_Subscription': paid_sub,
    'Traditional_Study_Hours': study_hours,
    'Perceived_AI_Dependency': ai_dep,
    'Anxiety_Level_During_Exams': anxiety,

    "Prompt_Engineering_Skill_Beginner":False,
    "Prompt_Engineering_Skill_Intermediate":False,
   

    # booleani caso d'uso inizializzati a false
    # 'Primary_Use_Case_Copywriting/Drafting': False,
    'Primary_Use_Case_Ideation': False,
    'Primary_Use_Case_Summarizing_Reading': False,
    'Primary_Use_Case_Debugging/Troubleshooting': False,
    'Primary_Use_Case_Direct_Answer_Generation': False,
    # inizializzazione policy istituzionali
    'Institutional_Policy_Strict_Ban': False,
    'Institutional_Policy_Allowed_With_Citation': False,
    # inizializzazione major bool
    'Major_Category_Business': False,
    'Major_Category_Medical': False,
    # 'Major_Category_Arts': False,
    'Major_Category_STEM': False,
    'Major_Category_Arts': False
}

# Attiviamo il flag corretto in base all'input selezionato dall'utente
    if prompt_skill ==0:
        features_comuni['Prompt_Engineering_Skill_Beginner']=True
    else:
        features_comuni['Prompt_Engineering_Skill_Intermediate']=True
    match use_case:
        case "Copywriting/Drafting":
            features_comuni['Primary_Use_Case_Copywriting/Drafting'] = True
        case "Ideation":
            features_comuni['Primary_Use_Case_Ideation'] = True
        case "Summarizing_Reading":
            features_comuni['Primary_Use_Case_Summarizing_Reading'] = True
        case "Coding":
            features_comuni['Primary_Use_Case_Debugging/Troubleshooting'] = True
        case "Research":
            features_comuni['Primary_Use_Case_Direct_Answer_Generation'] = True

    match policy:
        case "Allowed_With_Citation":
            features_comuni['Institutional_Policy_Allowed_With_Citation'] = True
        case "Strict_Ban":
            features_comuni['Institutional_Policy_Strict_Ban'] = True
        case "Actively_Encouraged":
            features_comuni['is_Actively_Encouraged'] = True

    match major:
        case "Humanities":
            features_comuni['Major_Category_Humanities'] = True
        case "Medical":
            features_comuni['Major_Category_Medical'] = True
        case "Business":
            features_comuni['Major_Category_Business'] = True
        case "STEM":
            features_comuni['Major_Category_STEM'] = True
        case "Arts":
            features_comuni['Major_Category_Arts'] = True
    return features_comuni


