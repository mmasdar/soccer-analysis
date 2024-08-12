import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

st.markdown("""
<style>
    .stSidebar {
        background-color: #f1f3f6;
        padding: 20px;
    }
    .stSidebar .sidebar-content {
        margin-top: 40px;
    }
    .sidebar-button {
        width: 100%;
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        background-color: #4CAF50;
        color: white;
        border: none;
        text-align: left;
        font-size: 16px;
    }
    .sidebar-button:hover {
        background-color: #45a049;
    }
    .search-input {
        width: 100%;
        padding: 10px;
        margin-top: 20px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv('clean_soccer.csv')

df = load_data()

with st.sidebar:
    st.image("logo.png", use_column_width=True)
    selected = option_menu("Soccer Dashboard", ['Player Analysis', 'Predict'], 
                            icons=['people-fill', 'magic'], 
                            menu_icon="soccer-ball", 
                            default_index=0)

col1, col2 = st.columns([3, 1])

def player_radar_chart(df, player_name, features):
    player_data = df[df['Name'] == player_name]
    if player_data.empty:
        st.error(f"Player '{player_name}' not found in the dataset.")
        return None
    
    all_players = df[df['Position'] == player_data['Position'].iloc[0]]
    avg_values = all_players[features].mean()
    player_values = player_data[features].iloc[0]
    
    min_values = all_players[features].min()
    max_values = all_players[features].max()
    normalized_avg = (avg_values - min_values) / (max_values - min_values)
    normalized_player = (player_values - min_values) / (max_values - min_values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_avg,
        theta=features,
        fill='toself',
        name='Average Player',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_player,
        theta=features,
        fill='toself',
        name=player_name,
        line=dict(color='red')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=f"Normalized Player Comparison: Average vs {player_name}",
        title_font_size=20,
        height=600,
        width=800
    )
    
    return fig, player_data

def predict_player_performance(player_data):
    positions = ['defender', 'midfielder', 'striker', 'goalkeeper']
    models = {}
    scalers = {}
    
    for position in positions:
        model_path = f'model/{position}_model.joblib'
        scaler_path = f'model/{position}_scaler.joblib'
        
        models[position] = joblib.load(model_path)
        scalers[position] = joblib.load(scaler_path)
    
    position = player_data['Position'].lower()
    
    if position not in positions:
        return "Invalid position"
    
    features = {
        'defender': ['Age', 'Appearances', 'Distance / 90 minutes', 'Interception'],
        'midfielder': ['Appearances', 'Key Passes', 'Pass Attempt / 90 minutes', 'Pass Completed / 90 minutes', 'Tackle Attempt', 'Tackle Won'],
        'striker': ['Appearances', 'Goals', 'Interception', 'Key Passes', 'Shots on Target'],
        'goalkeeper': ['Age', 'Appearances', 'Conceded', 'Shutouts']
    }
    
    X = [player_data[feature] for feature in features[position]]
    X_scaled = scalers[position].transform([X])
    
    prediction = models[position].predict(X_scaled)
    return prediction[0]

if selected == "Player Analysis":
    st.title("Dashboard Monitoring Player Performance")
    col1, col2 = st.columns([3, 1])

    with col1:
        player_name = st.selectbox("Select a player", df['Name'].unique())

        features = ['Distance / 90 minutes', 'Interception', 'Key Passes', 
                    'Pass Attempt / 90 minutes', 'Pass Completed / 90 minutes',
                    'Tackle Attempt', 'Tackle Won']

        fig, player_data = player_radar_chart(df, player_name, features)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


    with col2:
        if player_data is not None:
            st.subheader("Player Information")
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
                <h3 style='color: #1e3799; margin-bottom: 10px;'>{player_data['Name'].iloc[0]}</h3>
                <p><strong>Club:</strong> {player_data['Club'].iloc[0]}</p>
                <p><strong>Age:</strong> {player_data['Age'].iloc[0]} | <strong>Position:</strong> {player_data['Position'].iloc[0]}</p>
                <p><strong>Nationality:</strong> {player_data['Nationality'].iloc[0]}</p>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Key Stats")
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric(label="Appearances", value=player_data['Appearances'].iloc[0])
            with col2_2:
                st.metric(label="Goals", value=player_data['Goals'].iloc[0])
            with col2_3:
                st.metric(label="Assists", value=player_data['Assist'].iloc[0])

            performance = player_data['Performance'].iloc[0]
            if performance == 'Good':
                color = 'green'
            elif performance == 'Normal':
                color = 'blue'
            else:
                color = 'red'
            st.markdown(f"<h3 style='color: {color};'>Performance: {performance}</h3>", unsafe_allow_html=True)


elif selected == "Predict":
    st.title("Predict Player Performance")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=100, step=1)
        position = st.selectbox("Position", ["Striker", "Midfielder", "Defender", "Goalkeeper"])
        
        with st.expander("Key Stats"):
            appearances = st.number_input("Appearances", min_value=0, step=1)
            goals = st.number_input("Goals", min_value=0, step=1)
            assist = st.number_input("Assists", min_value=0, step=1)
    
    with col2:
        club = st.text_input("Club")
        nationality = st.text_input("Nationality")
        
        with st.expander("Detail Stats"):
            distance_90 = st.number_input("Distance / 90 minutes", min_value=0.0, step=0.1)
            interception = st.number_input("Interception", min_value=0, step=1)
            key_passes = st.number_input("Key Passes", min_value=0, step=1)
            pass_attempt_90 = st.number_input("Pass Attempt / 90 minutes", min_value=0.0, step=0.1)
            pass_completed_90 = st.number_input("Pass Completed / 90 minutes", min_value=0.0, step=0.1)
            tackle_attempt = st.number_input("Tackle Attempt", min_value=0, step=1)
            tackle_won = st.number_input("Tackle Won", min_value=0, step=1)
            shots_on_target = st.number_input("Shots on Target", min_value=0, step=1)
            conceded = st.number_input("Conceded", min_value=0, step=1)
            shutouts = st.number_input("Shutouts", min_value=0, step=1)

    if st.button("Predict Performance"):
        player_data = {
            'Position': position,
            'Age': age,
            'Appearances': appearances,
            'Goals': goals,
            'Interception': interception,
            'Key Passes': key_passes,
            'Shots on Target': shots_on_target,
            'Distance / 90 minutes': distance_90,
            'Pass Attempt / 90 minutes': pass_attempt_90,
            'Pass Completed / 90 minutes': pass_completed_90,
            'Tackle Attempt': tackle_attempt,
            'Tackle Won': tackle_won,
            'Conceded': conceded,
            'Shutouts': shutouts
        }
        
        predicted_performance = predict_player_performance(player_data)
        
        col_results1, col_results2 = st.columns([3, 2])
        with col_results1:
            st.subheader("Performance Radar Chart")
            features = ['Distance / 90 minutes', 'Interception', 'Key Passes', 
                        'Pass Attempt / 90 minutes', 'Pass Completed / 90 minutes',
                        'Tackle Attempt', 'Tackle Won']
            values = [player_data[feature] for feature in features]
            
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=features,
                fill='toself'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(values)]
                    )),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_results2:
            st.subheader("Player Summary")
            st.text(f"Name: {name}")
            st.text(f"Club: {club}")
            st.text(f"Age: {age} | Position: {position}")
            st.text(f"Nationality: {nationality}")
            st.text(f"Goals: {goals} | Assists: {assist}")
            
            st.subheader("Predicted Performance")
            st.markdown(f"<h3 style='color: {'green' if predicted_performance == 'Good' else 'blue' if predicted_performance == 'Normal' else 'red'};'>{predicted_performance}</h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    st.sidebar.text("Running main app prototype version created by mmasadar@gmail.com")