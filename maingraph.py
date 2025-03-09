import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import datetime

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

def plot_results(results):
    plt.figure(figsize=(10, 6))
    plt.bar(results['NAME'], results['UP_TIME'], label='Up Time')
    plt.bar(results['NAME'], results['DOWN_TIME'], bottom=results['UP_TIME'], label='Down Time')
    plt.xlabel('Name')
    plt.ylabel('Time (seconds)')
    plt.title('Up and Down Time Analysis')
    plt.legend()
    plt.show()

def main():
    file_path = "log.csv"
    df = load_and_process_data(file_path)
    
    if df is not None:
        results = calculate_times(df)
        print(results)
        plot_results(results)

if __name__ == "__main__":
    main()