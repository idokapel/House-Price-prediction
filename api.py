import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import os

def preprocessor(features):
    """
    :param features: The function gets a list of features (taken from html website) as an input -> assuming that all features are valid
    :return: df after initial preprocess - ready for the model
    """
    import pandas as pd
    import datetime
    columns = ['City', 'type', 'room_number', 'Area', 'Street', 'number_in_street',
               'city_area', 'num_of_images', 'floor_out_of', 'hasElevator',
               'hasParking', 'hasBars', 'hasStorage', 'condition', 'hasAirCondition',
               'hasBalcony', 'hasMamad', 'handicapFriendly', 'entranceDate',
               'furniture', 'publishedDays', 'description']
    df = pd.DataFrame(features, columns=columns)
    df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['num_of_images'] = pd.to_numeric(df['num_of_images'], errors='coerce')
    def get_floor(string):
        string = str(string)
        words = string.split()
        if len(words) > 2 and words[1] != 'קרקע':
            floor = int(string.split()[1])
        else:
            floor = 0
        return floor
    def get_total_floor(string):
        string = str(string)
        words = string.split()
        if len(words) > 2 and words[1] != 'קרקע':
            floor = int(string.split()[-1])
        else:
            floor = 0
        return floor
    df['total_floors'] = df['floor_out_of'].apply(get_total_floor)
    df['floor'] = df['floor_out_of'].apply(get_floor)
    def T_F(string):
        string = str(string)
        if string.isnumeric():
            return string
        if 'יש' in string:
            return 1
        elif 'כן' in string:
            return 1
        elif 'yes' in string.lower():
            return 1
        elif 'נגיש' in string:
            return 1
        elif 'true' in string.lower():
            return 1
        else:
            return 0
    for col in df.columns:
        if col.startswith('has'):
            df[col] = df[col].apply(T_F)
    df['handicapFriendly'] = df['handicapFriendly'].apply(T_F)
    def get_entrence_date(date):
        if 'מיידי' == date:
            result = 'less_than_6 months'
        elif 'לא צויין' == date:
            result = 'not_defined'
        if type(date) == str:
            result = "flexible"
        else:
            now = datetime.datetime.now()
            months = (date.year - now.year) * 12 + (date.month - now.month)
            if months < 6:
                result = "less_than_6 months "
            elif months < 12:
                result = "months_6_12"
            else:
                result = "above_year"
        return result
    df['entranceDate'] = df['entranceDate'].apply(get_entrence_date)
    feature_cols = ['City', 'type', 'room_number', 'Area', 'num_of_images',
                     'hasElevator', 'hasParking', 'hasBars', 'hasStorage', 'condition', 'hasAirCondition',
                     'hasBalcony', 'hasMamad', 'handicapFriendly', 'entranceDate', 'furniture',
                     'total_floors', 'floor']
    df = df[feature_cols]
    return df

app = Flask(__name__)
pipeline = pickle.load(open('trained_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')
    
    final_features = [features]
    
    df = preprocessor(final_features)
    price_pred = pipeline.predict(df)[0]
    output_text = '{:,.2f}'.format(price_pred)
    output_text = output_text + " שח "

    return render_template('index.html', prediction_text='{}'.format(output_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
