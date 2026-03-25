import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# Where a bird is recorded in the burrow in both years, this graph plots the probability of it returning.

def plot_return_prob(df, test_coll, DICT, folder_name):

    current_map = DICT['id_map_all']
    id_dict_swapped = {v: k for k, v in current_map.items()}
    mapped_df = df.rename(columns=id_dict_swapped)
    
    df_clean = mapped_df.dropna(how='all')
    returning_probs = df_clean.index.intersection(df_clean.columns)
    
    # Get probs for returning
    matching_list = [df_clean.loc[id, id] for id in returning_probs]
    matching_series = pd.Series(matching_list)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Plot return rate
    sns.ecdfplot(matching_series, ax=ax, color="red", linewidth=2, complementary=True)
    ax.set_ylabel("Return rate")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Plot probability desnity
    ax2 = ax.twinx()
    sns.kdeplot(matching_series, ax=ax2, fill=True, color="green", bw_adjust=0.1, clip=(0, 1))
    ax2.set_ylim(0)
    
    # X axis
    ax.set_xlim(-0.01, 1.01)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel("Probability threshold")
    graph_title = f"Probability of Return: {test_coll}"
    plt.title(graph_title)

    filename = f"return_probabilities-{test_coll}.png"
    filepath = folder_name / filename
    plt.savefig(filepath)
    print(f"Saved return probability graph to: {filepath}")
    
    plt.close()

# This sorts through the probabilities and gives us results!
# You can set two thresholds, one for returning birds and another for movements.
# A 'known' individual can only be predicted in one burrow
# It creates new audio_id for 'unknown' indivduals and generates an id map

def sort_predictions(df, RETURN_THRESHOLD, MOVEMENT_THRESHOLD, PAST_THRESHOLD, UNKNOWN, coll, DICT, prefix, pattern):

    current_map = DICT['id_map_all']
    id_dict_swapped = {v: k for k, v in current_map.items()}
    df_mapped = df.rename(columns=id_dict_swapped)
    
    # Making sure to drop empty rows in case
    df_clean = df_mapped.dropna(how='all')
    # Get's the ids for the trained classes, except for 'unkown' which get's treated differently
    y_columns = [c for c in df_clean.columns if c != UNKNOWN]

    results = {} 
    used_x = set()
    used_y = set()
    freed_columns = [] # To track columns where the 'home' bird didn't show up

    # if row and column ids match, check if the probability is above the set threshold
    for x_idx in df_clean.index:
        if x_idx in y_columns:
            prob = df_clean.loc[x_idx, x_idx]
            if pd.notna(prob) and prob >= RETURN_THRESHOLD:
                results[x_idx] = (x_idx, prob)
                used_x.add(x_idx)
                used_y.add(x_idx)
            else:
                # If a different bird appears in the burrow, the previous is assigned to a list
                freed_columns.append(x_idx)

    # looks to see if these 'replaced' birds are found elsewhere
    movement_matches = []
    for x_idx, row in df_clean[freed_columns].iterrows():
        if x_idx in used_x: continue
    
        for y_col, prob in row.items():
            if y_col not in used_y and pd.notna(prob) and prob >= MOVEMENT_THRESHOLD:
                movement_matches.append((prob, x_idx, y_col))

    # Sort by highest prob first
    movement_matches.sort(key=lambda x: x[0], reverse=True)

    for prob, x, y in movement_matches:
        if x not in used_x and y not in used_y:
            results[x] = (y, prob)
            used_x.add(x)
            used_y.add(y)

    # Checks other 'known' individuals from previous years for matches
    past_matches = []
    for x_idx, row in df_clean[y_columns].iterrows():
        if x_idx in used_x: continue  
    
        for y_col, prob in row.items():
            if y_col not in used_y and pd.notna(prob) and prob >= PAST_THRESHOLD:
                past_matches.append((prob, x_idx, y_col))

    past_matches.sort(key=lambda x: x[0], reverse=True)

    for prob, x, y in past_matches:
        if x not in used_x and y not in used_y:
            results[x] = (y, prob)
            used_x.add(x)
            used_y.add(y)

    # determines the rest as unknown
    for x in df_clean.index:
        if x not in used_x:
            unk_prob = df_clean.loc[x, UNKNOWN] if UNKNOWN in df_clean.columns else np.nan
            results[x] = (UNKNOWN, unk_prob)

    print("Predictions sorted by burrow")

    # creates a datafram with results and probabilities 
    final_df = pd.DataFrame.from_dict(results, orient='index', columns=['pred', 'prob']).reset_index().rename(columns={'index': 'individual_id'})
    current_map = DICT['id_map_all']
    final_df['pred'] = final_df['pred'].replace(current_map)

    extracted = final_df['individual_id'].str.extract(pattern)
    mask = final_df['pred'] == 'unknown'
    year_suffix = coll[-2:]
    final_df.loc[mask, 'pred'] = prefix + extracted.loc[mask, 0] + extracted.loc[mask, 1] + year_suffix

    new_map = pd.Series(final_df.pred.values,index=final_df.individual_id).to_dict()
    DICT[coll] = new_map

    print("Audio_ids generated for new birds")

    return final_df

def detailed_probabilities(df, test_coll, DICT, folder_name):
    
    prob_cols = df.columns.difference(['unknown'])

    def extract_rankings(row):
        top2 = row[prob_cols].nlargest(2)
        return pd.Series({
            'pred_1': top2.index[0],
            'prob_1': round(top2.values[0], 2),
            'pred_2': top2.index[1],
            'prob_2': round(top2.values[1], 2),
            'unknown': round(row['unknown'], 2)
        })
    
    result_prob = df.apply(extract_rankings, axis=1)
    result_prob = result_prob.reset_index()
    current_map = DICT[test_coll]
    result_prob['final_pred'] = result_prob['individual_id'].map(current_map)

    filename = f"{test_coll}_results.csv"
    filepath = folder_name / filename
    result_prob.to_csv(filepath, index=False)
    print(f"Saved detailed results to: {filepath}")

    return result_prob

def get_burrow_results(matrix, test_coll, COLL, DICT, index, folder_name):
    # result_cols remains the same (list of names)
    processed_coll = COLL[:index + 1][::-1]
    result_cols = [test_coll] + processed_coll
    burrow_results = pd.DataFrame(index=matrix.index, columns=result_cols)
    
    # 1. Map test_coll using its name as the key in DICT
    burrow_results[test_coll] = burrow_results.index.map(DICT.get(test_coll, {}))
    
    # 2. Map each processed collection using its name directly
    for col_name in processed_coll:
        # Instead of DICT[i], we use DICT[col_name]
        burrow_results[col_name] = burrow_results.index.map(DICT.get(col_name, {}))

    print(f"Burrow results created for: {', '.join(result_cols)}\n")

    filename = 'burrow_results.csv'
    filepath = folder_name / filename
    burrow_results.to_csv(filepath)
    print(f"Saved burrow results to: {filepath}")
    
    return burrow_results, result_cols

def get_combined_map(burrow_results, result_cols, DICT):
    
    # Prioritise most recent id map (first column)
    combined_series = burrow_results[result_cols[0]]
    
    # Append ids where missing from previous years/rounds
    for col in result_cols[1:]:
        combined_series = combined_series.combine_first(burrow_results[col])

    # Store it in your nested dictionary
    DICT['id_map_all'] = combined_series.dropna().to_dict()

    print(f"Combined maps for {', '.join(result_cols)}")

    # FIX: Access the length from the DICT key, not a variable
    print(f"Total individuals mapped: {len(DICT['id_map_all'])}")

def get_life_history(burrow_results, result_cols, folder_name):

    flat_values = burrow_results[result_cols].apply(lambda x: x.explode()).values.ravel()
    
    # 2. Filter out NaNs first
    clean_names = [name for name in flat_values if pd.notna(name)]
    
    # 3. Use set() to get unique values (it handles Python lists perfectly)
    unique_individuals = sorted(list(set(clean_names)))
    
    # 4. Create your new dataframe
    life_history = pd.DataFrame({'audio_id': unique_individuals})
    
    # checks if audio_id is in each collection
    for col in result_cols:
        life_history[col] = life_history['audio_id'].isin(burrow_results[col]).astype(int)

    filename = 'individual_results.csv'
    filepath = folder_name / filename
    life_history.to_csv(filepath, index=False)
    print(f"Saved life history results to: {filepath}")

    return life_history

