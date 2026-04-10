import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    
    # Load dataset
    df = pd.read_excel('../dataset/raw_data/crime_data_2001_2014.xlsx')

    # Rename columns
    df = df.rename(columns={
        'STATE/UT':'STATE',
        'KIDNAPPING & ABDUCTION':'KIDNAPPING',
        'DOWRY DEATHS':'DOWRY_DEATHS',
        'OTHER CRIMES':'OTHER_CRIMES',
        'TOTAL IPC CRIMES':'TOTAL_CRIMES'
    })

    # Remove missing & duplicates
    df = df.dropna()
    df = df.drop_duplicates()

    # Encoding
    state_encoder = LabelEncoder()
    district_encoder = LabelEncoder()

    df['STATE'] = state_encoder.fit_transform(df['STATE'])
    df['DISTRICT'] = district_encoder.fit_transform(df['DISTRICT'])

    # Save processed data
    df.to_csv('../dataset/processed_data/processed_crime_data.csv', index=False)

    # Save encoders
    pickle.dump(state_encoder, open('../outputs/results/state_encoder.pkl','wb'))
    pickle.dump(district_encoder, open('../outputs/results/district_encoder.pkl','wb'))

    print("Preprocessing completed")

if __name__ == "__main__":
    preprocess_data()