
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    # Load Data
    cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model_year', 'origin', 'car_name']
    df = pd.read_csv(filepath, names=cols, delim_whitespace=True)

    # Cleaning
    df['horsepower'] = df['horsepower'].replace('?', float('nan'))
    df['horsepower'] = df['horsepower'].astype(float)
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())
    df = df.drop(columns=['car_name'])

    return df

if __name__ == "__main__":
    df = load_and_preprocess('../data/auto-mpg.data')
    print("Data loaded successfully with shape:", df.shape)
