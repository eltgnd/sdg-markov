import streamlit as st
import pandas as pd
import plotly.express as px
from simulation import *

# Page config
st.set_page_config(
    page_title="Full Report",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Variables
labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
state_labels = list(labels.keys())
num_timesteps = 10

# Title
st.title('📊')
st.title('Full Report')

# Bar chart animation
def bar_animation(state):
    state_df = pd.DataFrame(state)
    state_df['Year'] = [2022+k for k in range(num_timesteps)]

    melted = pd.melt(state_df, 'Year', [i for i in range(1,18)], 'SDG')
    fig = px.bar(melted, x='SDG', y='value',
        animation_frame='Year',
        animation_group='SDG',
        width=650
    )
    fig.update_xaxes(tickmode='linear')
    fig.update_layout(yaxis_title='Probability')
    return fig

with st.container(border=True):
    st.caption('STATE PROBABILITIES OVER TIME')
    num_timesteps = st.number_input('Timestep', 10,30,10,1)

    # Read and create dataframes
    df = pd.read_csv('data.csv')

    prediction_matrices = {}
    for i in range(1,18):
        prediction_matrices[i] = markov_chain(df,i,num_timesteps)

    not_started, on_track, needs_acceleration, regresses = {}, {}, {}, {}
    for ind,state in enumerate([not_started, on_track, needs_acceleration, regresses]):
        for i in range(1,18):
            state[i] = {}
            for j in range(num_timesteps):
                state[i][j] = prediction_matrices[i][j][ind]
    
    state_dicts = [not_started, on_track, needs_acceleration, regresses]

    tab1, tab2, tab3, tab4 = st.tabs(state_labels)
    for ind,tab in enumerate([tab1,tab2,tab3,tab4]):
        with tab:
            st.plotly_chart(bar_animation(state_dicts[ind]))

# Correlation matrix
txt_file = open('correlation-matrix.txt', 'r')
corr_mat_str = txt_file.read()
corr_mat_str = corr_mat_str.replace('\n', ' ')
l = corr_mat_str.split(' ')

h = []
for i in l:
    if len(i) != 0 and i[0] in ['1', '0', '-']:
        h.append(float(i))
print(len(h) == 16**2) # Check if the formed list contains same number of elements

corr_mat = []
for i in range(16):
    corr_mat.append(h[i*16:(i+1)*16])
with st.expander('View Correlation Matrix'):
    st.write('Sourced at https://link.springer.com/article/10.1007/s11135-022-01443-4.')
    st.dataframe(corr_mat, hide_index=False)

# Bar chart net contribution correlation
corr_results = pd.read_csv('correlation_results.csv')
fig = px.bar(corr_results, x='Unnamed: 0', y=['Total Positive Contribution', 'Total Negative Contribution'],
    title='Contribution Comparison',
    barmode='group',
    width=650
)
fig.update_xaxes(tickmode='linear')
fig.update_layout(xaxis_title='SDG', yaxis_title='Contribution', legend={'x':0,'y':-0.4})

with st.container(border=True):
    st.plotly_chart(fig)