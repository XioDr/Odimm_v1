import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import pandas as pd
from datetime import datetime, timedelta, timezone

# --- Configuration ---
OANDA_ACCESS_TOKEN = "REDACTED"
OANDA_ACCOUNT_ID = "REDACTED"
INSTRUMENT = "XAU_USD"  # Gold/USD
GRANULARITY = "M1"      # 1-minute timeframe
MONTHS_OF_DATA = 1  # fetch 1 month (~30 days)

# --- API Connection ---
api = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN, environment="practice")

# --- Date Range Calculation (approx 30 days) ---
end_date_aware = datetime.now(timezone.utc)
start_date_aware = end_date_aware - timedelta(days=30 * MONTHS_OF_DATA)

# --- Data Fetching Loop ---
all_candles = []
current_start_dt = start_date_aware

print(f"Fetching ~{MONTHS_OF_DATA} month(s) of {INSTRUMENT} data in {GRANULARITY} timeframe...")

while current_start_dt < end_date_aware:
    from_time_str = current_start_dt.isoformat().replace('+00:00', 'Z')
    params = {
        "granularity": GRANULARITY,
        "from": from_time_str,
        "count": 5000
    }
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)

    try:
        api.request(r)
    except oandapyV20.exceptions.V20Error as err:
        print(f"OANDA API Error: {err}")
        break

    candles = r.response.get('candles', [])
    if not candles:
        print("No more data returned.")
        break

    all_candles.extend(candles)

    # last candle time
    last_candle_time_str = candles[-1]['time']
    last_candle_dt = pd.to_datetime(last_candle_time_str)
    current_start_dt = last_candle_dt + timedelta(minutes=1)

    if current_start_dt >= end_date_aware:
        print("Reached end date for requested range.")
        break

    print(f"Fetched up to: {last_candle_dt}")

# --- Data Processing and Saving ---
if all_candles:
    df = pd.DataFrame(all_candles)

    # normalize price columns
    if 'mid' in df.columns:
        df = df.join(pd.json_normalize(df['mid'].tolist())).drop(columns=['mid'])

    # rename and format columns
    df['time'] = pd.to_datetime(df['time'])
    df = df.rename(columns={
        'time': 'timestamp',
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'volume': 'Volume'
    })

    # drop duplicates and sort
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    # keep only the desired columns
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # save to CSV
    output_filename = f"{INSTRUMENT}_{GRANULARITY}_{MONTHS_OF_DATA}m.csv"
    df.to_csv(output_filename, index=False)

    print(f"\nData saved to '{output_filename}'")
    print(f"Total unique candles downloaded: {len(df)}")
    print(df.head())
else:
    print("\nCould not fetch any data.")
