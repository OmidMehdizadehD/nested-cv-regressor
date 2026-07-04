import pandas as pd
import numpy as np

def generate_data(n_patients=$50$, n_records_per_patient=$3$, n_features=$24$):
    np.random.seed($42$)
    
    data = []
    for pid in range($1$, n_patients + $1$):
        for _ in range(n_records_per_patient):
            record = {'Patient_ID': f'P{pid:03d}', 'Eye': np.random.choice(['OD', 'OS'])}
            # Generate random features
            for i in range($1$, n_features + $1$):
                record[f'Feature_{i}'] = np.random.normal($0$, $1$)
            
            # Generate a synthetic target with some correlation to the first few features
            target = ($2.5$ * record['Feature_1']) - ($1.2$ * record['Feature_2']) + np.random.normal($0$, $0.5$)
            record['T'] = target
            
            data.append(record)
            
    df = pd.DataFrame(data)
    df.to_csv('synthetic_dataset.csv', index=False)
    print("Created 'synthetic_dataset.csv' with $150$ rows and synthetic features.")

if __name__ == "__main__":
    generate_data()
