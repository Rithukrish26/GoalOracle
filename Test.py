import math

# ========================
# 📥 Input Functions
# ========================

def get_team_stats(team_name: str) -> dict:
    print(f"\n🔴 Enter stats for {team_name}:")
    return {
        "goals_scored": float(input("  Average goals scored per 90: ")),
        "goals_conceded": float(input("  Average goals conceded per 90: ")),
        "shots_on_target": int(input("  Shots on target per 90: ")),
        "chances_created": int(input("  Chances created per 90: ")),
        "possession": float(input("  Possession (%): ")),
        "pass_completion": float(input("  Pass completion (%): "))
    }

# ========================
# 📊 Display Functions
# ========================

def display_team_stats(team_name: str, stats: dict) -> None:
    print(f"\n📊 ─── {team_name.upper()} STATS ───")
    for key, value in stats.items():
        label = key.replace("_", " ").title()
        if key in ("possession", "pass_completion"):
            print(f"🔹 {label:<20}: {value:.1f}%")
        elif isinstance(value, float):
            print(f"🔹 {label:<20}: {value:.2f}")
        else:
            print(f"🔹 {label:<20}: {value}")

def display_score_probabilities(score_probs: dict, team_a_name: str, team_b_name: str, top_n: int = 10) -> None:
    print(f"\n📈 ─── SCORELINE PROBABILITIES: {team_a_name} vs {team_b_name} ───")
    print("🧮 Format: Team A - Team B\n")
    print(f"{'Scoreline':<10} {'Probability':>12}")
    print("-" * 24)
    for (a, b), prob in sorted(score_probs.items(), key=lambda x: -x[1])[:top_n]:
        print(f"{a}-{b:<7} {prob*100:>10.2f}%")

def display_outcome_probabilities(win_a: float, draw: float, win_b: float, team_a_name: str, team_b_name: str) -> None:
    print(f"\n🔮 ─── MATCH OUTCOME: {team_a_name.upper()} vs {team_b_name.upper()} ───")
    print(f"🏆 {team_a_name} Win: {win_a*100:.2f}%")
    print(f"🤝 Draw:           {draw*100:.2f}%")
    print(f"⚔️ {team_b_name} Win: {win_b*100:.2f}%")

# ========================
# ⚙️ Prediction Functions
# ========================

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function."""
    if lam < 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def predict_score_probabilities(lambda_a: float, lambda_b: float, max_goals: int = 6) -> dict:
    """
    Predict probability for each scoreline (a,b) where a,b in 0..max_goals using independent Poisson model.
    Returns a dict mapping (a,b) -> probability.
    """
    probs = {}
    for a in range(max_goals + 1):
        pa = poisson_pmf(a, lambda_a)
        for b in range(max_goals + 1):
            pb = poisson_pmf(b, lambda_b)
            probs[(a, b)] = pa * pb
    return probs

def calculate_outcomes(score_probs: dict) -> tuple:
    """Given scoreline probabilities, compute probabilities of Home win, Draw, Away win."""
    win_a = draw = win_b = 0.0
    for (a, b), p in score_probs.items():
        if a > b:
            win_a += p
        elif a == b:
            draw += p
        else:
            win_b += p
    return win_a, draw, win_b

# ========================
# ▶️ Main flow
# ========================

def main():
    # Get team names
    team_a_name = input("\n📝 Enter name for Team A: ").strip() or "Team A"
    team_b_name = input("📝 Enter name for Team B: ").strip() or "Team B"

    # Input stats
    team_a = get_team_stats(team_a_name)
    team_b = get_team_stats(team_b_name)

    # Display stats
    display_team_stats(team_a_name, team_a)
    display_team_stats(team_b_name, team_b)

    # Simple model: use average goals scored as lambda for Poisson
    # (You could refine this by adjusting for opponent's goals conceded.)
    lambda_a = team_a["goals_scored"]
    lambda_b = team_b["goals_scored"]

    # Optionally ask for max_goals to compute up to:
    try:
        max_goals = int(input("\n🔢 How many goals (max) to consider for each side? [default 11]: ") or 11)
    except ValueError:
        max_goals = 11

    # Predict score probabilities
    score_probs = predict_score_probabilities(lambda_a, lambda_b, max_goals=max_goals)
    display_score_probabilities(score_probs, team_a_name, team_b_name, top_n=10)

    # Outcome probabilities
    win_a, draw, win_b = calculate_outcomes(score_probs)
    display_outcome_probabilities(win_a, draw, win_b, team_a_name, team_b_name)


if __name__ == "__main__":
    main()
