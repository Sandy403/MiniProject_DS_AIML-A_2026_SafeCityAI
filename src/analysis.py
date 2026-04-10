import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def analyse():

    # Load model
    model = pickle.load(open('../outputs/results/rf_model.pkl','rb'))

    # Load encoders
    state_encoder = pickle.load(open('../outputs/results/state_encoder.pkl','rb'))
    district_encoder = pickle.load(open('../outputs/results/district_encoder.pkl','rb'))

    # Load KMeans
    kmeans = pickle.load(open('../outputs/results/kmeans.pkl','rb'))

    # ---------------------------
    # USER INPUT
    # ---------------------------
    state = input("Enter State: ")
    district = input("Enter District: ")
    year = int(input("Enter Year: "))

    # ---------------------------
    # ENCODE INPUT
    # ---------------------------
    try:
        state_encoded = state_encoder.transform([state])[0]
        district_encoded = district_encoder.transform([district])[0]
    except:
        print("Invalid State or District! Please enter correct values.")
        return

    # Create dataframe
    sample = pd.DataFrame({
        'STATE':[state_encoded],
        'DISTRICT':[district_encoded],
        'YEAR':[year]
    })

    # ---------------------------
    # PREDICTION
    # ---------------------------
    prediction = model.predict(sample)[0]

    crimes = ['MURDER','RAPE','KIDNAPPING','THEFT',
              'DOWRY_DEATHS','OTHER_CRIMES','TOTAL_CRIMES']

    # ---------------------------
    # HOTSPOT CLASSIFICATION
    # ---------------------------
    total_crime = prediction[6]
    cluster = kmeans.predict([[total_crime]])[0]

    cluster_labels = {0:'Low', 1:'Medium', 2:'High'}
    crime_level = cluster_labels[cluster]

    # ---------------------------
    # DISPLAY RESULTS
    # ---------------------------
    print("\nPredicted Crime Values:")
    for c, v in zip(crimes, prediction):
        print(f"{c}: {int(v)}")

    print("\nCrime Level:", crime_level)

    # ---------------------------
    # GRAPH
    # ---------------------------
    plt.figure(figsize=(10,5))
    ax = sns.barplot(x=crimes, y=prediction)

    # Add values on bars
    for i, v in enumerate(prediction):
        ax.text(i, v + max(prediction)*0.01, str(int(v)), ha='center')

    plt.title(f"Predicted Crimes ({crime_level} Risk Area)")
    plt.xticks(rotation=45)
    plt.show()


if __name__ == "__main__":
    analyse()