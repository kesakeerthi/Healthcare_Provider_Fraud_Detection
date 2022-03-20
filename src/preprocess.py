import pandas as pd

def func(InpatientClaim, ClaimDays, AdmittedDays):
    CD_Not_AD = 0
    if InpatientClaim==1 and (ClaimDays != AdmittedDays):
        CD_Not_AD = 1
    return CD_Not_AD

def preprocess(FinalData_Merge):
    '''This function takes details about a healthcare provider as input and preprocess it in a format required by model.'''

    # Missing/Null values of Numerical Columns in the dataset¶
    NullNumCols = ['AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt', 'DOB']
    print(FinalData_Merge['AdmissionDt'])
    FinalData_Merge[NullNumCols] = FinalData_Merge[NullNumCols].fillna(0)
    print(FinalData_Merge['AdmissionDt'])


    FinalData_Merge['ClmAdmitDiagnosisCode'] = FinalData_Merge['ClmAdmitDiagnosisCode'].apply(
        lambda x: 0 if pd.isna(x) else 1)

    FinalData_Merge['NoOfMonths_PartACov'] = FinalData_Merge['NoOfMonths_PartACov'].apply(lambda x: 0 if x == 0 else 1)
    FinalData_Merge['NoOfMonths_PartBCov'] = FinalData_Merge['NoOfMonths_PartBCov'].apply(lambda x: 0 if x == 0 else 1)

    FinalData_Merge['Gender'].replace([2, 1], [1, 0], inplace=True)

    FinalData_Merge['RenalDiseaseIndicator'].replace(['Y', '0'], [1, 0], inplace=True)
    FinalData_Merge['RenalDiseaseIndicator'] = pd.to_numeric(FinalData_Merge['RenalDiseaseIndicator'])

    ChronicConditions = ['ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
                         'ChronicCond_Cancer','ChronicCond_IschemicHeart','ChronicCond_stroke',
                         'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes',
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
       'CPT_Codes', 'PhysiciansCount', 'State_Encoded', 'County_Encoded',
       'Race_Encoded']

    FinalData_Merge = FinalData_Merge[ExpectedColOrder]

    return FinalData_Merge