import streamlit as st
import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="GoalOracle ⚽", layout="wide")

# --- Helper functions ------------------------------------------------------

def calculate_score_probabilities(lambda_a, lambda_b, max_goals=8):
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson.pmf(i, lambda_a) * poisson.pmf(j, lambda_b)
    return matrix

def calculate_outcome_probabilities(prob_matrix):
    win_a = np.tril(prob_matrix, -1).sum()
    draw = np.trace(prob_matrix)
    win_b = np.triu(prob_matrix, 1).sum()
    return win_a, draw, win_b

def most_probable_score(prob_matrix):
    idx = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
    return idx, prob_matrix[idx]

# --- Page UI ---------------------------------------------------------------

st.title("GoalOracle ⚽ — Predict")

# Centered logo with cream frame
try:
    logo = Image.open("Goal Oracle.png").convert("RGBA")
    size = (160, 160)
    logo = logo.resize(size)

    # Create a circular mask
    mask = Image.new("L", size, 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size[0], size[1]), fill=255)

    # Apply mask
    logo.putalpha(mask)

    # Add cream frame
    frame_size = 8
    frame_color = (255, 253, 208, 255)  # cream
    framed = Image.new("RGBA", (size[0] + frame_size * 2, size[1] + frame_size * 2), (0, 0, 0, 0))
    draw_frame = ImageDraw.Draw(framed)
    draw_frame.ellipse((0, 0, framed.size[0]-1, framed.size[1]-1), fill=frame_color)
    framed.paste(logo, (frame_size, frame_size), logo)

    st.markdown("<div style='display:flex; justify-content:center;'>", unsafe_allow_html=True)
    st.image(framed)
    st.markdown("</div>", unsafe_allow_html=True)
except Exception:
    st.write("")

st.markdown("---")

col1, col2, col3 = st.columns([1, 0.8, 1])

with col1:
    st.header("Team A — Inputs")
    ta_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.2, step=0.1, format="%.2f", key="ta_goals")
    ta_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.0, step=0.1)
    ta_sot = st.number_input("Shots on Target", min_value=0.0, value=3.0, step=0.1)
    ta_chances = st.number_input("Chances Created", min_value=0.0, value=5.0, step=0.1)
    ta_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=52.0, step=0.1)
    ta_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)

with col3:
    st.header("Team B — Inputs")
    tb_goals = st.number_input("Goals Scored (λ)", min_value=0.0, value=1.0, step=0.1, format="%.2f", key="tb_goals")
    tb_conceded = st.number_input("Goals Conceded", min_value=0.0, value=1.1, step=0.1)
    tb_sot = st.number_input("Shots on Target", min_value=0.0, value=2.7, step=0.1)
    tb_chances = st.number_input("Chances Created", min_value=0.0, value=4.0, step=0.1)
    tb_poss = st.number_input("Possession (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
    tb_pass = st.number_input("Pass Completion (%)", min_value=0.0, max_value=100.0, value=79.0, step=0.1)

with col2:
    st.write("\n" * 3)
    predict = st.button("Predict")
    reset = st.button("Reset")

# Reset behaviour
if reset:
    for k in ["ta_goals", "tb_goals"]:
        if k in st.session_state:
            st.session_state[k] = 0.0
    st.experimental_rerun()

# Default display
result_placeholder = st.empty()
heatmap_placeholder = st.empty()

if predict:
    try:
        lambda_a = float(ta_goals)
        lambda_b = float(tb_goals)
        if lambda_a < 0 or lambda_b < 0:
            raise ValueError("Lambdas must be non-negative")

        prob_matrix = calculate_score_probabilities(lambda_a, lambda_b, max_goals=8)
        win_a, draw, win_b = calculate_outcome_probabilities(prob_matrix)
        (best_i, best_j), best_p = most_probable_score(prob_matrix)

        with result_placeholder.container():
            st.subheader("Prediction Results")
            st.write(f"**Most Probable Score:** {best_i} - {best_j} ({best_p:.2%})")
            st.write(f"**Team A Win:** {win_a:.2%}   |   **Draw:** {draw:.2%}   |   **Team B Win:** {win_b:.2%}")
            st.markdown("---")
            st.write("**Input summary**")
            st.write({
                "Team A": {
                    "Goals Scored (λ)": lambda_a,
                    "Goals Conceded": ta_conceded,
                    "Shots on Target": ta_sot,
                    "Chances Created": ta_chances,
                    "Possession": ta_poss,
                    "Pass Completion": ta_pass,
                },
                "Team B": {
                    "Goals Scored (λ)": lambda_b,
                    "Goals Conceded": tb_conceded,
                    "Shots on Target": tb_sot,
                    "Chances Created": tb_chances,
                    "Possession": tb_poss,
                    "Pass Completion": tb_pass,
                }
            })

        with heatmap_placeholder.container():
            fig, ax = plt.subplots()
            im = ax.imshow(prob_matrix, origin='lower', aspect='auto')
            ax.set_xlabel('Team B Goals')
            ax.set_ylabel('Team A Goals')
            ax.set_title('Score Probability Matrix')
            for i in range(prob_matrix.shape[0]):
                for j in range(prob_matrix.shape[1]):
                    p = prob_matrix[i, j]
                    if p > 0.001:
                        ax.text(j, i, f"{p:.1%}", ha='center', va='center', fontsize=8)
            fig.colorbar(im, ax=ax)
            st.pyplot(fig)

        flat = []
        for i in range(prob_matrix.shape[0]):
            for j in range(prob_matrix.shape[1]):
                flat.append(((i, j), prob_matrix[i, j]))
        flat_sorted = sorted(flat, key=lambda x: x[1], reverse=True)
        top10 = [(f"{a}-{b}", f"{p:.2%}") for (a, b), p in flat_sorted[:10]]
        st.table(top10)

    except Exception as e:
        st.error(f"Invalid input detected: {e}")

st.markdown("---")
st.caption("GoalOracle — Poisson-based score prediction. Uses the 'Goals Scored' inputs as the Poisson λ for each team. This app preserves the original structure and formula from your Pygame project.")
