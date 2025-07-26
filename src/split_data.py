# src/split_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(df: pd.DataFrame, output_dir: str = "data", test_size: float = 0.2, random_state: int = 42):
    print("Filtering for main regions...")
    if 'league' not in df.columns:
        print("Missing 'league' column after processing. Cannot filter.")
        return

    # OPTIONAL: filter only major leagues (you can uncomment and adjust this)
    # df = df[df['league'].isin(['LCS', 'LEC', 'LCK', 'LPL'])]

    blue_team_idx = np.array(df['blue_team_idx'].tolist())
    red_team_idx = np.array(df['red_team_idx'].tolist())
    blue_champions_idx = np.array(df['blue_champions_idx'].tolist())
    red_champions_idx = np.array(df['red_champions_idx'].tolist())
    y = df['winner_is_blue'].values

    (bt_train, bt_test,
     rt_train, rt_test,
     bc_train, bc_test,
     rc_train, rc_test,
     y_train, y_test) = train_test_split(
        blue_team_idx, red_team_idx, blue_champions_idx, red_champions_idx, y,
        test_size=test_size, random_state=random_state
    )

    np.savez(f"{output_dir}/nn_train_split.npz",
             blue_team_idx=bt_train, red_team_idx=rt_train,
             blue_champions_idx=bc_train, red_champions_idx=rc_train,
             y=y_train)

    np.savez(f"{output_dir}/nn_test_split.npz",
             blue_team_idx=bt_test, red_team_idx=rt_test,
             blue_champions_idx=bc_test, red_champions_idx=rc_test,
             y=y_test)

    print("Data split and saved to data/nn_train_split.npz and data/nn_test_split.npz")
