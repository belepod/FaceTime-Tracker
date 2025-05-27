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

def convert_seconds_to_hours_minutes(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}h {minutes}m"

def impose_penalty(name, down_time_seconds):
    # Calculate penalty if down time is more than 1 hour
    if down_time_seconds > 3600:
        extra_minutes = (down_time_seconds - 3600) // 60
        penalty = extra_minutes * 10  # ₹10 per minute
        print(f"Penalty for {name}: Down time exceeded 1 hour. ₹{penalty} will be deducted.")

def calculate_times(df):
    results = []

    grouped = df.groupby('NAME')
    for name, group in grouped:
        group = group.sort_values('TIME')
        
        time_diff = group['TIME'].diff()
        
        up_time = time_diff[(group['STATUS'] == 'Exited') & (group['STATUS'].shift() == 'Entered')].sum()
        down_time = time_diff[(group['STATUS'] == 'Entered') & (group['STATUS'].shift() == 'Exited')].sum()
        
        up_time_seconds = up_time.total_seconds() if pd.notnull(up_time) else 0
        down_time_seconds = down_time.total_seconds() if pd.notnull(down_time) else 0
        
        # Imposing penalty if down time is greater than 1 hour
        impose_penalty(name, down_time_seconds)
        
        results.append({
            'NAME': name,
            'UP_TIME': convert_seconds_to_hours_minutes(up_time_seconds),
            'DOWN_TIME': convert_seconds_to_hours_minutes(down_time_seconds)
        })

    return pd.DataFrame(results)

def plot_results(results):
    results['UP_TIME_SEC'] = results['UP_TIME'].apply(lambda x: int(x.split('h')[0]) * 3600 + int(x.split('h')[1].split('m')[0]) * 60)
    results['DOWN_TIME_SEC'] = results['DOWN_TIME'].apply(lambda x: int(x.split('h')[0]) * 3600 + int(x.split('h')[1].split('m')[0]) * 60)
    
    plt.figure(figsize=(10, 6))
    plt.bar(results['NAME'], results['UP_TIME_SEC'], label='Up Time')
    plt.bar(results['NAME'], results['DOWN_TIME_SEC'], bottom=results['UP_TIME_SEC'], label='Down Time')
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
