afeCity AI – Crime Prediction and Hotspot Detection
This project was built to analyze crime patterns across India using real crime data from 2014 to 2022. The idea was simple — instead of manually going through crime records, we wanted a system that can automatically find which areas have high crime rates and also predict future crimes.

What the project does
We used two machine learning models:

K-Means Clustering to group regions into crime hotspots (Low / Medium / High)
Random Forest to predict how many crimes might happen in a given state, district, and year
You can enter a State, District, and Year — and the model will predict the number of murders, rapes, kidnappings, thefts, dowry deaths, and other crimes expected in that area.

Dataset
Indian Crime Dataset covering years 2014 to 2022. It contains crime records from different states and districts of India with details like crime type, location, and year.

How to run this project
Step 1 — Install the required libraries

pip install -r requirements.txt
Step 2 — Open Jupyter Notebook and run the notebooks in this order

data_understanding.ipynb — explore the dataset
preprocessing.ipynb — clean and prepare the data
visualization.ipynb — see the graphs and charts
Step 3 — To make predictions, run the analysis script

cd src
python analysis.py
Then enter the State, District, and Year when asked.

Libraries used
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
Model Results
R² Score: 88.89%
The model was able to correctly classify crime hotspot levels (Low/Medium/High) using K-Means
Team
Aditya Menon (Team Leader)
Santosh Kumar D
