import rosbag2_py
from rclpy.serialization import deserialize_message
from std_msgs.msg import Float64MultiArray
import pandas as pd
import numpy as np
import os

# ================== USER SETTINGS ==================
bag_path = '/home/kiyong/Downloads/rosbag2_2025_09_30-00_33_30'  # folder containing .db3
topics = [
    '/ekf/estimated_state',
    '/actuator_outputs',
    '/wind_data'
]
output_csv = 'combined_data.csv'
# ===================================================

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions('', '')
reader.open(storage_options, converter_options)

# Collect available topics
topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}

# Containers
data_dict = {t: [] for t in topics}

print(f"Reading bag: {bag_path}")
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic in topics:
        msg = deserialize_message(data, Float64MultiArray)
        data_dict[topic].append((t * 1e-9, list(msg.data)))  # timestamp in sec

# Convert to DataFrames
dfs = []
for topic, vals in data_dict.items():
    if not vals:
        print(f"⚠️ No messages for {topic}")
        continue
    df = pd.DataFrame(vals, columns=['time', 'data'])
    cols = df['data'].apply(pd.Series)
    cols.columns = [f"{topic}_{i}" for i in range(cols.shape[1])]
    df = pd.concat([df[['time']], cols], axis=1)
    dfs.append(df)

# Merge all on nearest timestamp
if not dfs:
    raise RuntimeError("No data found in the selected topics.")

df_merged = dfs[0]
for other in dfs[1:]:
    df_merged = pd.merge_asof(df_merged.sort_values('time'),
                              other.sort_values('time'),
                              on='time', direction='nearest', tolerance=0.05)  # 0.05s tolerance

# Save
df_merged.to_csv(output_csv, index=False)
print(f"✅ Saved combined CSV to {os.path.abspath(output_csv)}")
