import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(file_path):
    try:
        df = pd.read_csv(file_path)
        
        df['TIME'] = pd.to_datetime(df['TIME'], format='%H:%M:%S')
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File '{file_path}' is empty.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        return None

def calculate_times(df):
    results = []

    grouped = df.groupby('NAME')
    for name, group in grouped:
        group = group.sort_values('TIME')
        
        time_diff = group['TIME'].diff()
        
        up_time = time_diff[(group['STATUS'] == 'Exited') & (group['STATUS'].shift() == 'Entered')].sum()
        down_time = time_diff[(group['STATUS'] == 'Entered') & (group['STATUS'].shift() == 'Exited')].sum()
        
        results.append({'NAME': name, 'UP_TIME': up_time.total_seconds(), 'DOWN_TIME': down_time.total_seconds()})

    return pd.DataFrame(results)

def prepare_data_for_ml(results):
    X = results[['UP_TIME']]
    y = results['DOWN_TIME']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_and_train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    return model

def plot_results(results, model, scaler):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(results['NAME'], results['UP_TIME'], label='Up Time')
    plt.bar(results['NAME'], results['DOWN_TIME'], bottom=results['UP_TIME'], label='Down Time')
    plt.xlabel('Name')
    plt.ylabel('Time (seconds)')
    plt.title('Actual Up and Down Time')
    plt.legend()

    plt.subplot(1, 2, 2)
    up_times = results['UP_TIME'].values.reshape(-1, 1)
    up_times_scaled = scaler.transform(up_times)
    predicted_down_times = model.predict(up_times_scaled).flatten()

    plt.bar(results['NAME'], results['UP_TIME'], label='Up Time')
    plt.bar(results['NAME'], predicted_down_times, bottom=results['UP_TIME'], label='Predicted Down Time')
    plt.xlabel('Name')
    plt.ylabel('Time (seconds)')
    plt.title('Predicted Up and Down Time')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    file_path = "log.csv"
    df = load_and_process_data(file_path)
    
    if df is not None:
        results = calculate_times(df)
        print("Calculated times:")
        print(results)

        X_train, X_test, y_train, y_test, scaler = prepare_data_for_ml(results)
        model = create_and_train_model(X_train, y_train)

        test_loss = model.evaluate(X_test, y_test)
        print(f"\nTest Loss: {test_loss}")

        X_scaled = scaler.transform(results[['UP_TIME']])
        predictions = model.predict(X_scaled).flatten()
        results['PREDICTED_DOWN_TIME'] = predictions

        print("\nResults with predictions:")
        print(results)

        plot_results(results, model, scaler)

if __name__ == "__main__":
    main()