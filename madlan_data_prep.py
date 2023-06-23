
def prepare_data(df):
    """
    :param df: Raw DataFrame as input
    :return: Post-processed DataFrame - ready for Model training
    """
    import pandas as pd
    import numpy as np
    import re
    import datetime
    df.columns = df.columns.str.strip()

    # City
    df['City'] = df.City.str.strip()
    df['City'] = df.City.apply(lambda x: 'נהריה' if x == 'נהרייה' else x)

    # type
    df['type'] = df['type'].replace('אחר', 'דירה')

    # room_number
    def extract_room_number(string):
        string = str(string)
        room_num = ""
        for char in string:
            if char.isnumeric() or char == '.':
                room_num += char
        return room_num
    df['room_number'] = df['room_number'].apply(extract_room_number)
    df['room_number'] = pd.to_numeric(df['room_number'], errors='coerce')
    df['room_number'] = df['room_number'].fillna(df['room_number'].median())
    df['room_number'] = df['room_number'].replace(35, 3.5)

    # Area
    df['Area'] = df.Area.astype(str)
    df['Area'] = df.Area.apply(lambda x: "".join(re.findall('[0-9]', x)))
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['Area'].fillna(df['Area'].mean(), inplace=True)

    # Street
    def Str_cleaning(string):
        string = str(string)
        if len(string) <= 1:
            return np.nan
        else:
            string = re.sub(r'[^\w\s]', '', string)
            return string

    df['Street'] = df['Street'].apply(Str_cleaning)

    # city_area
    df['city_area'] = df['city_area'].apply(Str_cleaning)

    # price
    def get_numeric_price(price):
        if isinstance(price, str):
            match = re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d+)?', price)
            if match:
                price = int(match.group().replace(',', ''))
                return price
        elif isinstance(price, int):
            return price
    df.dropna(subset=['price'], inplace=True)
    df['price'] = df['price'].apply(get_numeric_price)
    df.dropna(subset=['price'], inplace=True)
    df = df[(df['price'] > 1000000) & (df['price'] < 10000000)].reset_index()


    # num_of_images
    df['num_of_images'] = df['num_of_images'].fillna(0)

    # floor_out_of
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

    # Binary cols
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

    # condition
    df['condition'] = df['condition'].replace('None', 'לא צויין')
    df['condition'] = df['condition'].replace(False, 'לא צויין')
    df['condition'] = df['condition'].fillna('לא צויין')

    # entranceDate
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

    # description
    df['description'] = df['description'].apply(Str_cleaning)
    df.drop_duplicates(inplace=True)

    return df



