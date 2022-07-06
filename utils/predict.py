import pandas as pd
from .preprocess import get_dataset_from_csv
from huggingface_hub import from_pretrained_keras

##Load Model
model = from_pretrained_keras("shivi/classification-grn-vsn")

def batch_predict(input_data):
    """
    This function is used for fetching predictions corresponding to input_dataframe.
    It outputs another dataframe containing: 
        1. prediction probability for each class
        2. actual expected outcome for each entry in the input dataframe
    """
    input_data_file = "input_data.csv"
    labels = ['Probability of Income greater than 50000',"Probability of Income less than 50000","Actual Income"]
    
    predictions_df = pd.DataFrame(columns=labels)

    input_data.to_csv(input_data_file, index=None, header=None)

    prod_dataset = get_dataset_from_csv(input_data_file, shuffle=True)
    
    pred = model.predict(prod_dataset)
    
    for prediction, actual_gt in zip(pred, input_data['income_level'].values.tolist()):
        y_pred_prob = round(prediction.flatten()[0] * 100, 2)
        y_not_prob = round((1-prediction.flatten()[0]) * 100, 2)
        y_pred = ">50000" if prediction.flatten()[0] > 0.5 else "<50000"
        prob_scores = {labels[0]: str(y_pred_prob)+"%" , labels[1]: str(y_not_prob)+"%", labels[2]: y_pred}
        predictions_df = predictions_df.append(prob_scores,ignore_index=True)
        
    return predictions_df


def user_input_predict(age, wage, cap_gains, cap_losses, dividends, num_persons, weeks_worked_in_year,
            class_of_worker, detailed_industry_recode,detailed_occupation_recode,education,
            enroll_in_edu_inst_last_wk, marital_stat, major_industry_code,major_occupation_code,
            race, hispanic_origin, sex, member_of_a_labor_union,reason_for_unemployment,
            full_or_part_time_employment_stat, tax_filer_stat,region_of_previous_residence,
            state_of_previous_residence,detailed_household_and_family_stat,detailed_household_summary_in_household,
            migration_codechange_in_msa,migration_codechange_in_reg, migration_codemove_within_reg,
            live_in_this_house_1_year_ago,migration_prev_res_in_sunbelt,family_members_under_18,
            country_of_birth_father,country_of_birth_mother,country_of_birth_self,
            citizenship,own_business_or_self_employed,fill_inc_questionnaire_for_veterans_admin,
            veterans_benefits, year):
    
    """
    This function is used for fetching model predictions based on inputs given by user on demo app
    """
    
    input_dict = {"age": [age],
    "class_of_worker": [class_of_worker],
    "detailed_industry_recode": [detailed_industry_recode],
    "detailed_occupation_recode": [detailed_occupation_recode],
    "education":[education],
    "wage_per_hour": [wage],
    "enroll_in_edu_inst_last_wk": [enroll_in_edu_inst_last_wk],
    "marital_stat": [marital_stat],
    "major_industry_code": [major_industry_code],
    "major_occupation_code": [major_occupation_code],
    "race": [race],
    "hispanic_origin": [hispanic_origin],
    "sex": [sex],
    "member_of_a_labor_union": [member_of_a_labor_union],
    "reason_for_unemployment": [reason_for_unemployment],
    "full_or_part_time_employment_stat": [full_or_part_time_employment_stat],
    "capital_gains": [cap_gains],
    "capital_losses": [cap_losses],
    "dividends_from_stocks": [dividends],
    "tax_filer_stat": [tax_filer_stat],
    "region_of_previous_residence": [region_of_previous_residence],
    "state_of_previous_residence": [state_of_previous_residence],
    "detailed_household_and_family_stat": [detailed_household_and_family_stat],
    "detailed_household_summary_in_household": [detailed_household_summary_in_household],
    "instance_weight": [0.0],
    "migration_code-change_in_msa": [migration_codechange_in_msa],
    "migration_code-change_in_reg": [migration_codechange_in_reg],
    "migration_code-move_within_reg": [migration_codemove_within_reg],
    "live_in_this_house_1_year_ago": [live_in_this_house_1_year_ago],
    "migration_prev_res_in_sunbelt": [migration_prev_res_in_sunbelt],
    "num_persons_worked_for_employer": [num_persons],
    "family_members_under_18": [family_members_under_18],
    "country_of_birth_father": [country_of_birth_father],
    "country_of_birth_mother": [country_of_birth_mother],
    "country_of_birth_self": [country_of_birth_self],
    "citizenship": [citizenship],
    "own_business_or_self_employed": [own_business_or_self_employed],
    "fill_inc_questionnaire_for_veterans_admin": [fill_inc_questionnaire_for_veterans_admin],
    "veterans_benefits": [veterans_benefits],
    "weeks_worked_in_year": [weeks_worked_in_year],
    "year": [year],
    "income_level": [0],
  }
    input_df = pd.DataFrame.from_dict(input_dict)
    input_data_file = "input_data.csv"
    
    input_df.to_csv(input_data_file, index=None, header=None)
    prod_dataset = get_dataset_from_csv(input_data_file, shuffle=True)
    
    labels = ['Income greater than 50000',"Income less than 50000"]
    prediction = model.predict(prod_dataset)
    y_pred_prob = round(prediction[0].flatten()[0],5)
    y_not_prob = round(1-prediction[0].flatten()[0],3)
    
    confidences = {labels[0]: float(y_pred_prob), labels[1]: float(y_not_prob)}
    return confidences
