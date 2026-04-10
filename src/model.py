import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def train_model():

    # Load processed data
    df = pd.read_csv('../dataset/processed_data/processed_crime_data.csv')

    # Features & Target
    X = df[['STATE','DISTRICT','YEAR']]
    y = df[['MURDER','RAPE','KIDNAPPING','THEFT',
            'DOWRY_DEATHS','OTHER_CRIMES','TOTAL_CRIMES']]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)
    print("Sample Prediction:", y_pred[0])

    # Evaluation
    r2 = r2_score(y_test, y_pred)
    print("R2 Score:", r2)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Model RMSE:", rmse)

    # Save model
    pickle.dump(model, open('../outputs/results/rf_model.pkl','wb'))

    # ---------------------------
    # KMeans Clustering
    # ---------------------------
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[['TOTAL_CRIMES']])

    pickle.dump(kmeans, open('../outputs/results/kmeans.pkl','wb'))

    print("Model training completed ✅")

if __name__ == "__main__":
    train_model()