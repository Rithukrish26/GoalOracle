import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

# ==========================
# COLOR PALETTE & STYLE
# ==========================
BLACK = "#0A0A0A"
GOLD = "#D4AF37"
BEIGE = "#F5F5DC"
CREAM = "#FFFDD0"

# Custom streamlit CSS for background and form
st.markdown(f"""
  <style>
    .stApp {{
      background: linear-gradient(135deg, {BLACK} 60%, {CREAM} 100%);
    }}
    .block-bg {{
      background: {BEIGE};
      border-radius: 20px;
      padding: 2rem 2rem 1rem 2rem;
      box-shadow: 0 4px 32px 0 {BLACK}77;
    }}
    h1, h2, .stTextInput > label, .stButton button {{
      color: {GOLD};
    }}
    .minimalist {{
      background-image:
        repeating-linear-gradient(30deg, {GOLD}22 0px, {GOLD}22 2px, transparent 2px, transparent 80px),
        radial-gradient(circle at 80% 10%, {CREAM}33 0, transparent 60%),
        radial-gradient(circle at 10% 75%, {BEIGE}44 0, transparent 60%);
      background-repeat: no-repeat;
      background-size: cover;
    }}
  </style>
""", unsafe_allow_html=True)

st.markdown('<div class="minimalist"></div>', unsafe_allow_html=True)

# ==========================
# Page Title & Description
# ==========================
st.markdown("<h1 style='text-align: center;'>GoalOracle ⚽</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #D4AF37;'>Football Score & Outcome Predictor</h3>", unsafe_allow_html=True)
st.write("")

# ==========================
# INPUT BLOCK
# ==========================
st.markdown('<div class="block-bg">', unsafe_allow_html=True)
st.subheader("Enter Match Statistics:")

colA, colB = st.columns(2)

with colA:
    st.markdown("#### Team A")
    goals_scored_a = st.number_input("Average goals scored per 90", min_value=0.0, step=0.1, format="%.2f", value=1.2)
    goals_conceded_a = st.number_input("Average goals conceded per 90", min_value=0.0, step=0.1, format="%.2f", value=1.1)
    shots_a = st.number_input("Shots on target per 90", min_value=0, value=5)
    chances_a = st.number_input("Chances created per 90", min_value=0, value=8)
    possession_a = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=52.5)
    pass_a = st.number_input("Pass completion (%)", min_value=0.0, max_value=100.0, value=83.4)

with colB:
    st.markdown("#### Team B")
    goals_scored_b = st.number_input("Average goals scored per 90  ", min_value=0.0, step=0.1, format="%.2f", value=1.0)
    goals_conceded_b = st.number_input("Average goals conceded per 90  ", min_value=0.0, step=0.1, format="%.2f", value=1.3)
    shots_b = st.number_input("Shots on target per 90  ", min_value=0, value=6)
    chances_b = st.number_input("Chances created per 90  ", min_value=0, value=7)
    possession_b = st.number_input("Possession (%) ", min_value=0.0, max_value=100.0, value=47.5)
    pass_b = st.number_input("Pass completion (%) ", min_value=0.0, max_value=100.0, value=79.2)
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# ==========================
# Calculate & Display Results
# ==========================
def calculate_score_probabilities(lambda_a, lambda_b, max_goals=5):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for goals_a in range(max_goals + 1):
        for goals_b in range(max_goals + 1):
            matrix[goals_a][goals_b] = poisson.pmf(goals_a, lambda_a) * poisson.pmf(goals_b, lambda_b)
    return matrix

def calculate_outcome_probabilities(prob_matrix):
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    return win_a, draw, win_b

if st.button("🔮 Predict!"):
    # Use attacking strengths (lambda) for score prediction
    lambda_a = goals_scored_a
    lambda_b = goals_scored_b

    # Score probabilities matrix
    prob_matrix = calculate_score_probabilities(lambda_a, lambda_b)

    # Display outcome summary
    win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
    st.markdown(f"<h4 style='color:{GOLD};'>Match Outcome Probabilities</h4>", unsafe_allow_html=True)
    st.write(f"🏆 Team A Win: **{win_a:.2%}**")
    st.write(f"🤝 Draw: **{draw:.2%}**")
    st.write(f"⚔ Team B Win: **{win_b:.2%}**")

    # Show heatmap
    st.write("")
    st.markdown(f"<h4 style='color:{GOLD};'>Score Probability Heatmap</h4>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8,6))
    plt.imshow(prob_matrix, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="Probability")
    plt.xlabel("Team B Goals", color=BLACK)
    plt.ylabel("Team A Goals", color=BLACK)
    ax.set_xticks(np.arange(prob_matrix.shape[1]))
    ax.set_yticks(np.arange(prob_matrix.shape[0]))
    for i in range(prob_matrix.shape[0]):
        for j in range(prob_matrix.shape[1]):
            ax.text(j, i, f"{prob_matrix[i, j]:.1%}", ha='center', va='center', color='black', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)

