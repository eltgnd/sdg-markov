# Import packages
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
from simulation import *

# Page config
st.set_page_config(
    page_title="SDG+Markov",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Simulation variables
labels = {'Not Started':0, 'On Track':1, 'Needs Acceleration':2, 'Regresses':3}
state_labels = list(labels.keys())
n = len(labels)
current_year = 2023
convergence_val = 30

# Helper functions
def list_to_latex_matrix(lst):
    latex_matrix = "\\begin{bmatrix}"
    for row in lst:
        if isinstance(row, list):  # check if row is a list
            for num in row:
                latex_matrix += str(num) + " & "
            latex_matrix = latex_matrix[:-2]  # remove the last "&"
        else:  # if row is not a list, it's an integer
            latex_matrix += str(row) + " & "
            latex_matrix = latex_matrix[:-2]  # remove the last "&"
        latex_matrix += "\\\\"
    latex_matrix += "\\end{bmatrix}"
    return latex_matrix

# Session states
ss = st.session_state
initial_variables = {'simulate_button':False}
for var,val in initial_variables.items():
    if var not in ss:
        ss[var] = val

def activate(var, val):
    ss[var] = val

# Title
st.image('https://github.com/eltgnd/sdg-markov/blob/master/logo.png?raw=true',width=100)
st.title('SDG+Markov')
with st.container(border=True):
    st.write('**Driving Sustainability Forward: Southeast Asia\'s Sustainable Development Goals Progression through Markovian Correlations**')
    st.write('This project conducts a Markov chain simulation on the progress states of the United Nations-initiated agenda, the 2030 Sustainable Development Goals (SDGs). After the simulation, correlations between each SDG\'s results were analyzed using a correlation SDG matrix by Pakkan et. al (2022).')
    st.link_button('View Google Colab Code', 'https://colab.research.google.com/drive/1rj9HwqAaJbhn-PMFno5ZcDcTDBho1VUY')

with st.expander('Research Objectives'):
    st.write("""

- Examine the interrelationships between each Sustainable Development Goal (SDG) to understand their correlations.
- Assess the feasibility of achieving the Sustainable Development Goals (SDGs) by 2030 in the context of the Asia-Pacific region, considering their intercorrelations.
- Determine the optimal overarching direction and strategic approach, including policy recommendations, for the Asia-Pacific region to attain the Sustainable Development Goals (SDGs) by 2030.
    """)

st.caption('INTERACTIVE SIMULATION')

# User Input
with st.form('simulation'):
    sdg = st.selectbox('Select an SDG',
        """SDG 1: No poverty
        SDG 2: Zero hunger
        SDG 3: Good health and well-being
        SDG 4: Quality education
        SDG 5: Gender equality
        SDG 6: Clean water and sanitation
        SDG 7: Affordable and clean energy
        SDG 8: Decent work and economic growth
        SDG 9: Industry, innovation, and infrastructure
        SDG 10: Reduced inequalities
        SDG 11: Sustainable cities and communities
        SDG 12: Responsible consumption and production
        SDG 13: Climate action
        SDG 14: Life below water
        SDG 15: Life on land
        SDG 16: Peace, justice, and strong institutions""".split('\n'),
        help='Disclaimer: SDG 17 is not included due to limited data.',)
    t = st.number_input('Timestep', 1,50,1,1, help='Timestep refers to number of years to predict from 2023.')

    s = st.form_submit_button("Simulate")
if s:
    activate('simulate_button', s)
    
if ss.simulate_button:
    # Simulation
    df = pd.read_csv('data.csv')

    # Set-up
    sdg_number = int(sdg.split()[1][:-1])
    st.caption(f'SET-UP FOR SDG {sdg_number}')

    initial_mat = create_initial_state_vector(df, sdg_number, current_year)
    trans_mat = create_transition_matrix(convert_sdg_states(df, sdg_number))
    with st.container(border=True):
        st.write('\n')
        col1, col2, col3 = st.columns([0.5,0.7,0.2])
        with col1:
            st.write(f"""Initial state vector $=$ ${list_to_latex_matrix(initial_mat)}$""")
        with col2:
            st.write(f"""Transition matrix $=$ ${list_to_latex_matrix(trans_mat)}$""")
        st.write('\n')

    # Markov chain result
    st.caption('MARKOV CHAIN RESULT')

    result = markov_chain(df, sdg_number, t)
    result['State'] = state_labels
    result_df = pd.DataFrame(result)

    cols = result_df.columns.tolist()
    result_df = result_df[cols[-1:] + cols[:-1]]
    column_labels = {key:value for key,value in zip([i for i in range(len(result_df.keys()))],[2023+i for i in range(len(result_df.keys()))])}

    result_df = result_df.rename(columns=column_labels)
    st.dataframe(result_df, use_container_width =True, hide_index=True)

    # Convergent
    st.write(f'''As $k\\rightarrow\infty$, the current model provides the following probabilities of each state:''')

    convergence_test = markov_chain(df, sdg_number, convergence_val)
    col1, col2, col3, col4 = st.columns(4)
    for ind,col in enumerate([col1, col2, col3, col4]):
        col.metric(state_labels[ind], convergence_test[convergence_val-1][ind])
    style_metric_cards(box_shadow=False, border_left_color='#D3D3D3')

    # Graph
    with st.container(border=True):
        headers = result_df.transpose().iloc[0]
        result_graph  = pd.DataFrame(result_df.transpose().values[1:], columns=headers)
        result_graph['Year'] = [2023+i for i in range(t)]
        fig = px.line(result_graph, x='Year', y=state_labels,
            title= f'SDG {sdg_number} State Probabilities for {t} Years',
            markers=True,
            width=650
        )
        fig.update_layout(legend={'x':0,'y':-1.0}) 
        st.plotly_chart(fig)

    # Correlation
    df_pos = pd.read_csv('positive_correlation.csv')
    df_neg = pd.read_csv('negative_correlation.csv')

    with st.container(border=True):
        tab1, tab2 = st.tabs(['Positive Correlation Breakdown', 'Negative Correlation Breakdown'])
        with tab1:
            fig = px.bar(df_pos, x=[i+1 for i in range(16)], y=str(sdg_number),
                title=f'Positive Contribution of SDG {sdg_number} to Other SDGs',
                width=650
            )
            fig.update_layout(xaxis_title='SDG', yaxis_title='Contribution')
            st.plotly_chart(fig)
        with tab2:
            fig = px.bar(df_neg, x=[i+1 for i in range(16)], y=str(sdg_number),
                title=f'Negative Contribution of SDG {sdg_number} to Other SDGs',
                width=650
            )
            fig.update_layout(xaxis_title='SDG', yaxis_title='Contribution')
            st.plotly_chart(fig)
