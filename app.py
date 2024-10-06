from flask import Flask, jsonify, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

# Load weather data from CSV
data = pd.read_csv("weather_data_2019_2024.csv", parse_dates=['Date'])
data['year'] = data['Date'].dt.year
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day

# Features and labels for the model
X = data[['year', 'month', 'day']]
y = data[['Temperature', 'Humidity', 'Prec', 'Pressure', 'WindSpeed']]

# Train the model (RandomForest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Precipitation type mapping based on the index
precipitation_types = {
    0: 'None',
    1: 'Rain',
    2: 'Rain and snow mixed',
    3: 'Snow',
    4: 'Sleet (US definition)',
    5: 'Freezing rain',
    6: 'Hail'
}

# Modified prediction function for the next N days with precipitation type
def predict_weather(start_date, days=15):
    future_dates = pd.date_range(start=start_date, periods=days)
    future_df = pd.DataFrame({
        'year': future_dates.year,
        'month': future_dates.month,
        'day': future_dates.day
    })
    
    predictions = model.predict(future_df)
    
    # Rounding numerical predictions
    prediction_df = pd.DataFrame(predictions, columns=['Temperature', 'Humidity', 'Precipitation', 'Pressure', 'WindSpeed'],
                                 index=future_dates)
    prediction_df = prediction_df.round(0)
    
    # Adding formatted date without time
    prediction_df['Date'] = future_dates.strftime('%a, %d %b %Y')

    # Mapping the Precipitation values to the types
    # prediction_df['Precipitation'] = prediction_df['Precipitation'].apply(lambda x: precipitation_types.get(int(x), 'Unknown'))
    print(prediction_df)
    return prediction_df[['Date', 'Temperature', 'Humidity', 'Precipitation', 'Pressure', 'WindSpeed']]


# API to get weather forecast for the next N days
@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    # Get start date from query param or use today's date
    start_date_str = request.args.get('start_date')
    if start_date_str:
        start_date = pd.to_datetime(start_date_str)
    else:
        start_date = date.today()

    # Predict weather for the next 15 days
    predictions = predict_weather(start_date)
    predictions_list = predictions.to_dict(orient='records')

    # Return JSON response
    return jsonify(predictions_list)

# API to get forecast for a specific date
@app.route('/api/forecast/<date_str>', methods=['GET'])
def get_forecast_for_specific_date(date_str):
    try:
        specific_date = pd.to_datetime(date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format."}), 400
    
    # Predict weather for the next 15 days starting from the specific date
    predictions = predict_weather(specific_date, days=1)
    predictions_list = predictions.to_dict(orient='records')
    
    return jsonify(predictions_list)

# API to get current weather (example values)
@app.route('/api/current', methods=['GET'])
def get_current_weather():
    # You can fetch real-time data from a weather API or simulate it
    current_weather = {
        'Date': date.today().strftime('%Y-%m-%d'),
        'Temperature': 25.0,  # Example value
        'Humidity': 60,       # Example value
        'Precipitation': 0,   # Example value
        'Pressure': 1015,     # Example value
        'WindSpeed': 15       # Example value
    }
    return jsonify(current_weather)

# API to get historical weather data
@app.route('/api/historical', methods=['GET'])
def get_historical_weather():
    date_str = request.args.get('date')

    if not date_str:
        return jsonify({"error": "Date is required."}), 400

    try:
        historical_date = pd.to_datetime(date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format."}), 400

    historical_data = data[data['Date'] == historical_date]
    
    if historical_data.empty:
        return jsonify({"error": "No historical data available for this date."}), 404

    return jsonify(historical_data.to_dict(orient='records'))

# API to get average weather for a date range
@app.route('/api/average', methods=['GET'])
def get_average_weather():
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Start and end dates are required."}), 400

    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format."}), 400

    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    if filtered_data.empty:
        return jsonify({"error": "No data available for the specified date range."}), 404

    average_data = filtered_data.mean().to_dict()
    average_data['Date'] = f"{start_date.date()} to {end_date.date()}"

    return jsonify(average_data)

# API to get mean weather forecast for the next year
@app.route('/api/mean_forecast', methods=['GET'])
def get_mean_forecast():
    # Get start date from query param or use today's date
    start_date_str = request.args.get('start_date')
    if start_date_str:
        start_date = pd.to_datetime(start_date_str)
    else:
        start_date = date.today()

    # Predict weather for the next 365 days
    predictions = predict_weather(start_date, days=365)
    
    # Convert Date to datetime for accessing month/year
    predictions['Date'] = pd.to_datetime(predictions['Date'])
    
    # Group by month and calculate mean values
    predictions['month'] = predictions['Date'].dt.month
    mean_predictions = predictions.groupby('month').mean().reset_index()

    # Return the mean predictions as JSON
    return jsonify(mean_predictions.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
