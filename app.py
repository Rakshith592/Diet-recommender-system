from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)



# Absolute path example
file_path = 'IndianFoods.csv'
foodfinal = pd.read_csv(file_path)

foodfinal.head()


# Normalize Numerical Features
scaler = StandardScaler()
X_numerical = scaler.fit_transform(foodfinal[['Calories', 'Carbohydrates', 'Protein', 'Fats', 'Fibre', 'Sodium', 'Calcium','Iron']])


#Integer linear programming

from pulp import *
import pandas as pd
def bmi_calculator(height, weight):
    bmi = weight/(height**2)
    if bmi<18.5:
        return [bmi, "Underweight"]
    elif bmi>18.5 and bmi<24.9:
        return [bmi, "Normal weight"]
    else:
        return [bmi,"Overweight"]

def tdee_calculator(weight, height, age, gender, activity):
    if gender==0:
        bmr = 10*weight+625*height-5*age+5
    elif gender==1:
        bmr = 10*weight+625*height-5*age-161
    else:
        calories = bmr*activity
    return bmr*activity

nutrient_constraints = {
    'Sodium': {'max': 2000},         # mg       
    'Fibre': {'max': 30}           # mg
}

def generate_meal_plan_lp(meals_df, daily_calories, max_meals=3, nutrient_constraints=nutrient_constraints):
    # Create the problem
    prob = LpProblem("MealPlanning", LpMinimize)

    # Macronutrient targets
    macronutrient_ratios = {'carbs': 0.6, 'protein': 0.15, 'fat': 0.25}
    carb_target = (daily_calories * macronutrient_ratios['carbs']) / 4
    protein_target = (daily_calories * macronutrient_ratios['protein']) / 4
    fat_target = (daily_calories * macronutrient_ratios['fat']) / 9

    # Create a binary variable for each meal
    meal_vars = LpVariable.dicts("Meal", meals_df.index, cat="Binary")
    
    # Constraint: total meals selected ≤ max_meals
    prob += lpSum([meal_vars[i] for i in meals_df.index]) <= max_meals

    # Objective: minimize deviation from target macros (soft objective)
    prob += lpSum([
        (meals_df.loc[i, 'Calories'] - daily_calories / max_meals) ** 2 * meal_vars[i] +
        (meals_df.loc[i, 'Carbohydrates'] - carb_target / max_meals) ** 2 * meal_vars[i] +
        (meals_df.loc[i, 'Protein'] - protein_target / max_meals) ** 2 * meal_vars[i] +
        (meals_df.loc[i, 'Fats'] - fat_target / max_meals) ** 2 * meal_vars[i]
        for i in meals_df.index
    ])

    # Caloric constraints (95%-105%)
    total_calories = lpSum([meals_df.loc[i, 'Calories'] * meal_vars[i] for i in meals_df.index])
    prob += total_calories >= daily_calories * 0.99
    prob += total_calories <= daily_calories * 1.01

    # Optional: Macronutrient range constraints (±10%)
    total_carbs = lpSum([meals_df.loc[i, 'Carbohydrates'] * meal_vars[i] for i in meals_df.index])
    total_protein = lpSum([meals_df.loc[i, 'Protein'] * meal_vars[i] for i in meals_df.index])
    total_fat = lpSum([meals_df.loc[i, 'Fats'] * meal_vars[i] for i in meals_df.index])

    prob += total_carbs >= carb_target * 0.9
    prob += total_carbs <= carb_target * 1.1

    prob += total_protein >= protein_target * 0.9
    prob += total_protein <= protein_target * 1.1

    prob += total_fat >= fat_target * 0.9
    prob += total_fat <= fat_target * 1.1

    if nutrient_constraints:
        for nutrient, limits in nutrient_constraints.items():
            total_nutrient = lpSum(meals_df.loc[i, nutrient] * meal_vars[i] for i in meals_df.index)
            if 'min' in limits:
                prob += total_nutrient >= limits['min'], f"{nutrient}_min"
            if 'max' in limits:
                prob += total_nutrient <= limits['max'], f"{nutrient}_max"
                
    # Solve the problem
    prob.solve()

    # Get selected meals
    selected_meals = meals_df.loc[[i for i in meals_df.index if meal_vars[i].value() == 1]]

    return selected_meals



height = 1.78
weight = 58
BMI = bmi_calculator(height, weight)
print(BMI)
age = 22
gender = 0
activity = 1.2
tdee = tdee_calculator(weight, height, age, gender, activity)
if BMI[0]<18.5:
    calories = tdee+500
elif BMI[0]>24.9:
    calories = tdee-500
else:
    calories = tdee
print(calories)
recommendations = generate_meal_plan_lp(foodfinal, calories, max_meals=3, nutrient_constraints=nutrient_constraints)
print(recommendations)
recommendations=recommendations.to_dict(orient='records')
for meal in recommendations:
    print(meal['Dish_Name'])
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form = request.form
        height = float(form['height'])
        weight = float(form['weight'])
        BMI = bmi_calculator(height, weight)

        # Initialize defaults
        recommendations = []

        # Proceed only if all TDEE-related inputs are provided
        if form.get('age') and form.get('gender') and form.get('activity'):
            age = float(form['age'])
            gender = float(form['gender'])
            activity = float(form['activity'])

            tdee = tdee_calculator(weight, height, age, gender, activity)

            # Adjust calories based on BMI
            if BMI[0] < 18.5:
                calories = tdee + 500
            elif BMI[0] > 24.9:
                calories = tdee - 500
            else:
                calories = tdee

            # Generate meal recommendations
            recommendations = generate_meal_plan_lp(foodfinal,
                calories,
                max_meals=3,
                nutrient_constraints=nutrient_constraints
            ).to_dict(orient='records')

        return render_template('index.html', BMI=BMI, recommendations=recommendations)

    return render_template('index.html', BMI=[], recommendations=[])

if __name__ == '__main__':
        port = int(os.environ.get("PORT", 5000)) # Default to 5000 for local testing
        app.run(debug=True, host='0.0.0.0', port=port)
