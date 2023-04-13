import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df['overweight'] = [1 if x > 25 else 0 for x in df['weight']/((df['height']/100)**2)]

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df[['cholesterol', 'gluc']] = df[['cholesterol', 'gluc']].applymap(lambda x: 0 if x == 1 else 1)

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.groupby('cardio', as_index=False).value_counts().rename(columns={'count':'total'})
    
    # Draw the catplot with 'sns.catplot()'
    g = sns.catplot(data=df_cat, x='variable', y='total', kind='bar', col='cardio', hue='value', order=['active','alco','cholesterol', 'gluc', 'overweight', 'smoke']);

    # Get the figure for the output
    fig = g.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    # Clean the data
    
    h_less = df['height'].quantile(0.025)
    h_more = df['height'].quantile(0.975)
    w_less = df['weight'].quantile(0.025)
    w_more = df['weight'].quantile(0.975)
    
    # filter using quantiles
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & (df['height'] >= h_less) & (df['height'] <= h_more) & (df['weight'] >= w_less) & (df['weight'] <= w_more)]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(18, 8))

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidth=.5);

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
