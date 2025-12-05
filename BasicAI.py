import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Create sample data
data = {'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
        'Exam_Score': [10, 20, 30, 40, 50, 60, 70, 80, 85, 95]}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df)


plt.scatter(df['Study_Hours'], df['Exam_Score'], color='blue')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score")
plt.show()


X = df[['Study_Hours']]  # Features (Input)
y = df['Exam_Score']  # Target (Output)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create & train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict scores for test data
y_pred = model.predict(X_test)

# Compare actual vs predicted
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

error = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {error}")

hours_studied = [[7.8]]  # Input: 7.5 hours of study
predicted_score = model.predict(hours_studied)
print(f"Predicted Exam Score for 7.8 hours of study: {predicted_score[0]:.2f}")



#python --version version 3.7 or later
#pip install pandas matplotlib scikit-learn
#python -c 'import pandas, matplotlib, sklearn; print("All libraries installed successfully")'

#for Ubuntu Create and Activate a Virtual Environment (Recommended) 
#python3 -m venv myenv
#source myenv/bin/activate

#pip install pandas matplotlib scikit-learn



