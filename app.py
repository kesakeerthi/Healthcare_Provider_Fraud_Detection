import numpy as np
import pandas as pd
import streamlit as st
import pickle


# Loading Standard Scaler model to scale the data
scaler = pickle.load(open('C:/Users/Keerthi/Desktop/Healthcare-Fraud-Detection/pickles/standard_scaler.pkl', 'rb'))

# Loading Random Forest which is trained and tuned with Dataset
model = pickle.load(open('C:/Users/Keerthi/Desktop/Healthcare-Fraud-Detection/pickles/xg_boost.pkl', 'rb'))

# Loading Encoding Dicts used while training for State, County, Race
County_Encoded = pickle.load(open('C:/Users/Keerthi/Desktop/Healthcare-Fraud-Detection/pickles/County_Encoded.pkl', 'rb'))
State_Encoded = pickle.load(open('C:/Users/Keerthi/Desktop/Healthcare-Fraud-Detection/pickles/State_Encoded.pkl', 'rb'))
Race_Encoded = pickle.load(open('C:/Users/Keerthi/Desktop/Healthcare-Fraud-Detection/pickles/Race_Encoded.pkl', 'rb'))

def func(InpatientClaim, ClaimDays, AdmittedDays):
    CD_Not_AD = 0
    if InpatientClaim==1 and (ClaimDays != AdmittedDays):
        CD_Not_AD = 1
    return CD_Not_AD


def predict_fraud(X):
    '''This function takes details about a healthcare provider as input and returns a prediction of the healthcare provider
       being a potential fraud.'''

    # Scaling data
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)

    return y_pred, y_prob


def preprocess(FinalData_Merge):
    '''This function takes details about a healthcare provider as input and preprocess it in a format required by model.'''

    # Missing/Null values of Numerical Columns in the dataset¶
    NullNumCols = ['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt', 'DOB', 'DeductibleAmtPaid']
    FinalData_Merge[NullNumCols] = FinalData_Merge[NullNumCols].fillna(0)

    FinalData_Merge['ClmAdmitDiagnosisCode'] = FinalData_Merge['ClmAdmitDiagnosisCode'].apply(
        lambda x: 0 if pd.isna(x) else 1)

    FinalData_Merge['NoOfMonths_PartACov'] = FinalData_Merge['NoOfMonths_PartACov'].apply(lambda x: 0 if x == 0 else 1)
    FinalData_Merge['NoOfMonths_PartBCov'] = FinalData_Merge['NoOfMonths_PartBCov'].apply(lambda x: 0 if x == 0 else 1)

    FinalData_Merge['Gender'].replace([2, 1], [1, 0], inplace=True)

    FinalData_Merge['RenalDiseaseIndicator'].replace(['Y', '0'], [1, 0], inplace=True)

    ChronicConditions = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
                         'ChronicCond_Cancer',
                         'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
                         'ChronicCond_IschemicHeart',
                         'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis']
    for col in ChronicConditions:
        FinalData_Merge[col].replace([1, 2], [0, 1], inplace=True)

    FinalData_Merge['isDead'] = FinalData_Merge['DOD'].apply(lambda x: 0 if pd.isnull(x) else 1)

    FinalData_Merge['AdmissionDt'] = pd.to_datetime(FinalData_Merge['AdmissionDt'])
    FinalData_Merge['DischargeDt'] = pd.to_datetime(FinalData_Merge['DischargeDt'])
    FinalData_Merge['AdmittedDays'] = (FinalData_Merge['DischargeDt'] - FinalData_Merge['AdmissionDt']).apply(
        lambda x: x.days)

    FinalData_Merge['ClaimStartDt'] = pd.to_datetime(FinalData_Merge['ClaimStartDt'])
    FinalData_Merge['ClaimEndDt'] = pd.to_datetime(FinalData_Merge['ClaimEndDt'])
    FinalData_Merge['DOB'] = pd.to_datetime(FinalData_Merge['DOB'])
    FinalData_Merge['Age'] = FinalData_Merge['ClaimStartDt'].dt.year - FinalData_Merge['DOB'].dt.year

    FinalData_Merge['ClaimDays'] = (FinalData_Merge['ClaimEndDt'] - FinalData_Merge['ClaimStartDt']).apply(
        lambda x: x.days)

    FinalData_Merge['isCD_not_AD'] = FinalData_Merge.apply(
        lambda x: func(x.InpatientClaim, x.ClaimDays, x.AdmittedDays), axis=1)

    ICD_Codes = ['ClmDiagnosisCode_1', 'ClmDiagnosisCode_2', 'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                 'ClmDiagnosisCode_5',
                 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8', 'ClmDiagnosisCode_9',
                 'ClmDiagnosisCode_10']
    FinalData_Merge['ICD_Codes'] = FinalData_Merge[ICD_Codes].apply(lambda x: sum(pd.notnull(x)), axis=1)

    CPT_Codes = ['ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
                 'ClmProcedureCode_4', 'ClmProcedureCode_5', 'ClmProcedureCode_6']
    FinalData_Merge['CPT_Codes'] = FinalData_Merge[CPT_Codes].apply(lambda x: sum(pd.notnull(x)), axis=1)

    FinalData_Merge['DeductibleAmtPaid'] = FinalData_Merge['DeductibleAmtPaid'].apply(lambda x: 0 if x == 0 else 1)

    FinalData_Merge['AttendingPhysician'] = FinalData_Merge['AttendingPhysician'].apply(
        lambda x: 0 if pd.isnull(x) else 1)
    FinalData_Merge['OperatingPhysician'] = FinalData_Merge['OperatingPhysician'].apply(
        lambda x: 0 if pd.isnull(x) else 1)
    FinalData_Merge['OtherPhysician'] = FinalData_Merge['OtherPhysician'].apply(lambda x: 0 if pd.isnull(x) else 1)

    FinalData_Merge['PhysiciansCount'] = FinalData_Merge[
        ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']].apply(lambda x: sum(x), axis=1)



    Cols_tobe_Dropped = ['Provider', 'BeneID', 'ClaimID', 'DiagnosisGroupCode', 'DOD', 'AdmissionDt', 'DischargeDt',
                         'DOB', 'ClaimStartDt', 'ClaimEndDt', 'ClmDiagnosisCode_1', 'ClmDiagnosisCode_2',
                         'ClmDiagnosisCode_3', 'ClmDiagnosisCode_4',
                         'ClmDiagnosisCode_5', 'ClmDiagnosisCode_6', 'ClmDiagnosisCode_7', 'ClmDiagnosisCode_8',
                         'ClmDiagnosisCode_9',
                         'ClmDiagnosisCode_10', 'ClmProcedureCode_1', 'ClmProcedureCode_2', 'ClmProcedureCode_3',
                         'ClmProcedureCode_4',
                         'ClmProcedureCode_5', 'ClmProcedureCode_6', 'State', 'County', 'Race']

    FinalData_Merge.drop(Cols_tobe_Dropped, axis='columns', inplace=True)

    ExpectedColOrder = ['InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician',
       'OtherPhysician', 'ClmAdmitDiagnosisCode', 'DeductibleAmtPaid',
       'InpatientClaim', 'Gender', 'RenalDiseaseIndicator',
       'NoOfMonths_PartACov', 'NoOfMonths_PartBCov', 'ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',
       'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt',
       'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt', 'isDead',
       'AdmittedDays', 'Age', 'ClaimDays', 'isCD_not_AD', 'ICD_Codes',
       'CPT_Codes', 'PhysiciansCount', 'Race_Encoded', 'State_Encoded',
       'County_Encoded']

    FinalData_Merge = FinalData_Merge[ExpectedColOrder]

    return FinalData_Merge


def main():
    html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">Healthcare Provider Fraud Predictor </h2>
    </div>
    <br>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("Please enter the following details of the claim to know if related provider has done fraud or not.")
    st.write("")

    # Example Values
    Provider = 'PRV12345'
    BeneID = 'BENE1234'
    ClaimID = 'CLM123456'

    col1, col2 = st.columns(2)

    with col1:
        Provider = st.text_input("Provider Id", help='Eg PRV12345', max_chars=8)
        BeneID = st.text_input("Patient/Beneficiary Id", help='Eg BENE12345', max_chars=9)
        Race = st.number_input("Race", min_value=1, max_value=5)
        State = st.number_input("State", help='1-54',min_value=1, max_value=54)
        County = st.number_input("County", min_value=0, max_value=999)
        DOB = st.text_input("Date of Birth", help='Eg 2009-04-12')
        DOD = st.text_input("Date of Death", help='Eg 2009-04-12')
        Gender = st.number_input("Gender", min_value=1, max_value=2)
        NoOfMonths_PartACov = st.number_input("Part A Coverage in Months", min_value=0, max_value=12)
        NoOfMonths_PartBCov = st.number_input("Part B Coverage in Months", min_value=0, max_value=12)
        IPAnnualReimbursementAmt = st.number_input("InPatient AnnualReimbursementAmount", min_value=0)
        IPAnnualDeductibleAmt = st.number_input("InPatient AnnualDeductibleAmount", min_value=0)
        OPAnnualReimbursementAmt = st.number_input("OutPatient AnnualReimbursementAmount", min_value=0)
        OPAnnualDeductibleAmt = st.number_input("OutPatient AnnualDeductibleAmount", min_value=0)
        RenalDiseaseIndicator = st.text_input("Renal Disease Indicator", max_chars=1, help='0/Y')
        ChronicCond_Alzheimer = st.number_input("Chronic Condition Alzheimer", min_value=1, max_value=2)
        ChronicCond_Heartfailure = st.number_input("Chronic Condition Heartfailure", min_value=1, max_value=2)
        ChronicCond_KidneyDisease = st.number_input("Chronic Condition KidneyDisease", min_value=1, max_value=2)
        ChronicCond_Cancer = st.number_input("Chronic Condition Cancer", min_value=1, max_value=2)
        ChronicCond_ObstrPulmonary = st.number_input("Chronic Condition Obstrutive Pulmonary", min_value=1, max_value=2)
        ChronicCond_Depression = st.number_input("Chronic Condition Depression", min_value=1, max_value=2)
        ChronicCond_Diabetes = st.number_input("Chronic Condition Diabetes", min_value=1, max_value=2)
        ChronicCond_IschemicHeart = st.number_input("Chronic Condition IschemicHeart", min_value=1, max_value=2)
        ChronicCond_Osteoporasis = st.number_input("Chronic Condition Osteoporasis", min_value=1, max_value=2)
        ChronicCond_rheumatoidarthritis = st.number_input("Chronic Condition rheumatoidarthritis", min_value=1,
                                                          max_value=2)
        ChronicCond_stroke = st.number_input("Chronic Condition stroke", min_value=1, max_value=2)
        ClmProcedureCode_4 = st.number_input("Claim Procedure Code 4", min_value=3848, max_value=9999)

    with col2:
        ClaimID = st.text_input("Claim Id", help='Eg CLM34146, CLM162739', max_chars=9)
        ClmAdmitDiagnosisCode = st.text_input("Claim Admit Diagnosis Code", help='Eg 5849/71848/E9305')
        AdmissionDt = st.text_input("Admission Date", help='Eg 2009-04-12')
        DischargeDt = st.text_input("Discharge Date", help='Eg 2009-04-12')
        InpatientClaim = st.number_input("InpatientClaim", min_value=0, max_value=1)
        ClaimStartDt = st.text_input("Claim Start Date", help='Eg 2009-04-12')
        ClaimEndDt = st.text_input("Claim End Date", help='Eg 2009-04-12')
        InscClaimAmtReimbursed = st.number_input("Insurance Claim Amount Reimbursed", min_value=0)
        AttendingPhysician = st.text_input("Attending Physician", help='Eg PHY365867')
        OperatingPhysician = st.text_input("Operating Physician", help='Eg PHY365867')
        OtherPhysician = st.text_input("Other Physician", help='Eg PHY365867')

        DeductibleAmtPaid = st.number_input("Deductible Amount Paid", min_value=0)

        DiagnosisGroupCode = st.text_input("Diagnosis GroupCode", help='Eg 5849/71848/E9305')
        ClmDiagnosisCode_1 = st.text_input("Claim Diagnosis Code 1", help='Eg 5849/71848/E9305')
        ClmDiagnosisCode_2 = st.text_input("Claim Diagnosis Code 2", help='Eg 5849/71848/E9305')
        ClmDiagnosisCode_3 = st.text_input("Claim Diagnosis Code 3", "")
        ClmDiagnosisCode_4 = st.text_input("Claim Diagnosis Code 4", "")
        ClmDiagnosisCode_5 = st.text_input("Claim Diagnosis Code 5", "")
        ClmDiagnosisCode_6 = st.text_input("Claim Diagnosis Code 6", "")
        ClmDiagnosisCode_7 = st.text_input("Claim Diagnosis Code 7", "")
        ClmDiagnosisCode_8 = st.text_input("Claim Diagnosis Code 8", "")
        ClmDiagnosisCode_9 = st.text_input("Claim Diagnosis Code 9", "")
        ClmDiagnosisCode_10 = st.text_input("Claim Diagnosis Code 10", "")
        ClmProcedureCode_1 = st.number_input("Claim Procedure Code 1", min_value=3848, max_value=9999)
        ClmProcedureCode_2 = st.number_input("Claim Procedure Code 2", min_value=3848, max_value=9999)
        ClmProcedureCode_3 = st.number_input("Claim Procedure Code 3", min_value=3848, max_value=9999)
        ClmProcedureCode_5 = st.number_input("Claim Procedure Code 5", min_value=3848, max_value=9999)
        ClmProcedureCode_6 = st.number_input("Claim Procedure Code 6", min_value=3848, max_value=9999)


    FinalData_Merge = pd.DataFrame({"BeneID": BeneID, "ClaimID": ClaimID, "ClaimStartDt": ClaimStartDt,
                                    "ClaimEndDt": ClaimEndDt, "InscClaimAmtReimbursed": InscClaimAmtReimbursed,
                                    "AttendingPhysician": AttendingPhysician, "OperatingPhysician": OperatingPhysician,
                                    "OtherPhysician": OtherPhysician, "AdmissionDt": AdmissionDt,
                                    "ClmAdmitDiagnosisCode": ClmAdmitDiagnosisCode,"Provider": Provider,
                                    "DeductibleAmtPaid": DeductibleAmtPaid,
                                    "DischargeDt": DischargeDt, "DiagnosisGroupCode": DiagnosisGroupCode,
                                    "ClmDiagnosisCode_1": ClmDiagnosisCode_1, "ClmDiagnosisCode_2": ClmDiagnosisCode_2,
                                    "ClmDiagnosisCode_3": ClmDiagnosisCode_3, "ClmDiagnosisCode_4": ClmDiagnosisCode_4,
                                    "ClmDiagnosisCode_5": ClmDiagnosisCode_5, "ClmDiagnosisCode_6": ClmDiagnosisCode_6,
                                    "ClmDiagnosisCode_7": ClmDiagnosisCode_7, "ClmDiagnosisCode_8": ClmDiagnosisCode_8,
                                    "ClmDiagnosisCode_9": ClmDiagnosisCode_9,
                                    "ClmDiagnosisCode_10": ClmDiagnosisCode_10,
                                    "ClmProcedureCode_1": ClmProcedureCode_1, "ClmProcedureCode_2": ClmProcedureCode_2,
                                    "ClmProcedureCode_3": ClmProcedureCode_3, "ClmProcedureCode_4": ClmProcedureCode_4,
                                    "ClmProcedureCode_5": ClmProcedureCode_5, "ClmProcedureCode_6": ClmProcedureCode_6,
                                    "InpatientClaim": InpatientClaim, "DOB": DOB, "DOD": DOD, "Gender": Gender,
                                    "Race": Race, "State": State, "County": County,
                                    "NoOfMonths_PartACov": NoOfMonths_PartACov,
                                    "NoOfMonths_PartBCov": NoOfMonths_PartBCov,
                                    "RenalDiseaseIndicator": RenalDiseaseIndicator,
                                    "ChronicCond_Alzheimer": ChronicCond_Alzheimer,
                                    "ChronicCond_Heartfailure": ChronicCond_Heartfailure,
                                    "ChronicCond_KidneyDisease": ChronicCond_KidneyDisease,
                                    "ChronicCond_Cancer": ChronicCond_Cancer,
                                    "ChronicCond_ObstrPulmonary": ChronicCond_ObstrPulmonary,
                                    "ChronicCond_Depression": ChronicCond_Depression,
                                    "ChronicCond_Diabetes": ChronicCond_Diabetes,
                                    "ChronicCond_IschemicHeart": ChronicCond_IschemicHeart,
                                    "ChronicCond_Osteoporasis": ChronicCond_Osteoporasis,
                                    "ChronicCond_stroke": ChronicCond_stroke,
                                    "IPAnnualReimbursementAmt": IPAnnualReimbursementAmt,
                                    "IPAnnualDeductibleAmt": IPAnnualDeductibleAmt,
                                    "OPAnnualReimbursementAmt": OPAnnualReimbursementAmt,
                                    "OPAnnualDeductibleAmt": OPAnnualDeductibleAmt,
                                    "ChronicCond_rheumatoidarthritis": ChronicCond_rheumatoidarthritis},index=[0])

    FinalData_Merge['Race_Encoded'] = Race_Encoded.get(Race)
    FinalData_Merge['State_Encoded'] = State_Encoded.get(State)
    FinalData_Merge['County_Encoded'] = County_Encoded.get(County)


    X = preprocess(FinalData_Merge)
    input_array = np.array(X).reshape(1, -1)
    
    res, prob = "", ""
    if st.button("Predict"):
        y_pred, y_prob =predict_fraud(input_array)
        if y_pred == 1:
            res = 'Fraud'
            prob = round(y_prob[:, 1][0]*100,2)
        else:
            res = 'Not Fraud'
            prob = round(y_prob[:, 0][0]*100,2)
            
        st.write('Health Care Provider with ID {} is {} with probability {}%'\
               .format(Provider, res, prob))


if __name__ == '__main__' :
    main()