
import nflreadpy as nfl
import pandas as pd

# Load historical NFL data
stats = nfl.load_player_stats([2024, 2025]).to_pandas()

stats = stats[stats["season_type"] == "REG"].copy()

# Calculate total team carries each week
team_carries = (
    stats.groupby(["season", "team", "week"])["carries"]
    .sum()
    .reset_index()
)

team_carries = team_carries.rename(
    columns={"carries": "team_carries"}
)

# Keep running backs
rb = stats[stats["position"] == "RB"].copy()

rb = rb[[
    "player_id",
    "player_name",
    "season",
    "team",
    "week",
    "carries",
    "rushing_yards"
]].copy()

# Add team carries
rb = rb.merge(
    team_carries,
    on=["season", "team", "week"],
    how="left"
)

rb["carry_share"] = (
    rb["carries"] / rb["team_carries"].replace(0, float("nan"))
) * 100


# Same percentage change function as our model
def percent_change(recent, baseline):
    if baseline <= 0:
        return 0

    change = ((recent - baseline) / baseline) * 100

    return max(-100, min(100, change))


results = []

# Test each player separately for each season
for (season, player_id), games in rb.groupby(
    ["season", "player_id"]
):

    games = games.sort_values("week").reset_index(drop=True)

    # Need 4 previous games plus 3 future games
    if len(games) < 7:
        continue

    # Simulate making predictions throughout the season
    for i in range(4, len(games) - 2):

        past = games.iloc[:i]

        recent = past.tail(3)
        baseline = past.iloc[:-3]

        future = games.iloc[i:i+3]

        # Historical averages
        recent_carries = recent["carries"].mean()
        baseline_carries = baseline["carries"].mean()

        recent_yards = recent["rushing_yards"].mean()
        baseline_yards = baseline["rushing_yards"].mean()

        recent_share = recent["carry_share"].mean()
        baseline_share = baseline["carry_share"].mean()

        # YPC
        recent_attempts = recent["carries"].sum()
        baseline_attempts = baseline["carries"].sum()

        recent_ypc = (
            recent["rushing_yards"].sum() / recent_attempts
            if recent_attempts > 0 else 0
        )

        baseline_ypc = (
            baseline["rushing_yards"].sum() / baseline_attempts
            if baseline_attempts > 0 else 0
        )

        # Percentage changes
        carries_change = percent_change(
            recent_carries, baseline_carries
        )

        yards_change = percent_change(
            recent_yards, baseline_yards
        )

        share_change = percent_change(
            recent_share, baseline_share
        )

        ypc_change = percent_change(
            recent_ypc, baseline_ypc
        )

        # Future Outlook Score
        score = (
            0.35 * carries_change
            + 0.35 * share_change
            + 0.15 * yards_change
            + 0.15 * ypc_change
        )

        # Workload reliability
        workload = (recent_carries + baseline_carries) / 2
        reliability = min(1, workload / 12)

        score = score * reliability

        # Actual performance in next 3 games
        future_yards = future["rushing_yards"].mean()
        future_carries = future["carries"].mean()

        # Compare future production to recent production
        yards_improved = future_yards > recent_yards
        carries_improved = future_carries > recent_carries

        results.append({
            "season": season,
            "player": games["player_name"].iloc[0],
            "week": int(past["week"].iloc[-1]),
            "future_score": score,
            "recent_yards": recent_yards,
            "future_yards": future_yards,
            "recent_carries": recent_carries,
            "future_carries": future_carries,
            "yards_improved": yards_improved,
            "carries_improved": carries_improved
        })


backtest = pd.DataFrame(results)

# Classify model predictions
def classify(score):
    if score >= 20:
        return "STRONG IMPROVEMENT"
    elif score >= 5:
        return "LIKELY IMPROVING"
    elif score > -5:
        return "STABLE"
    elif score > -20:
        return "LIKELY DECLINING"
    else:
        return "STRONG DECLINE"


backtest["prediction"] = backtest["future_score"].apply(
    classify
)

# Summarize results
summary = backtest.groupby("prediction").agg(
    predictions=("future_score", "count"),
    avg_score=("future_score", "mean"),
    yards_improved_pct=("yards_improved", "mean"),
    carries_improved_pct=("carries_improved", "mean")
)

summary["yards_improved_pct"] *= 100
summary["carries_improved_pct"] *= 100

print("\nRB FUTURE OUTLOOK BACKTEST")
print("Total predictions:", len(backtest))

print("\nRESULTS BY PREDICTION CATEGORY")
print(summary.round(2).to_string())

# Overall improvement rates for comparison
print("\nOVERALL BASELINE")
print(
    "Yards improved:",
    round(backtest["yards_improved"].mean() * 100, 2), "%"
)

print(
    "Carries improved:",
    round(backtest["carries_improved"].mean() * 100, 2), "%"
)
