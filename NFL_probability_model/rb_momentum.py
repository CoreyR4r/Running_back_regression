
import nflreadpy as nfl
import pandas as pd

# Load weekly player stats
stats = nfl.load_player_stats([2026]).to_pandas()

# Only regular season games
stats = stats[stats["season_type"] == "REG"]


# Calculate team rushing attempts each week

team_carries = (
    stats.groupby(["team", "week"])["carries"]
    .sum()
    .reset_index()
)

team_carries = team_carries.rename(
    columns={"carries": "team_carries"}
)


# Keep running backs
rb = stats[stats["position"] == "RB"].copy()

# Keep the statistics we need
rb = rb[[
    "player_id",
    "player_name",
    "team",
    "week",
    "carries",
    "rushing_yards"
]].copy()

# Calculate yards per carry
rb["ypc"] = rb["rushing_yards"] / rb["carries"].replace(0, float("nan"))

# Sort by player and week
rb = rb.sort_values(["player_id", "week"])


# Add total team carries to each running back
rb = rb.merge(
    team_carries,
    on=["team", "week"],
    how="left"
)

# Calculate each player's share of team carries
rb["carry_share"] = (
    rb["carries"] / rb["team_carries"].replace(0, float("nan"))
) * 100


print(rb.head(20))
print("Total rows:", len(rb))

# STEP 2: Calculate recent and baseline averages

results = []

for player_id, games in rb.groupby("player_id"):

    games = games.sort_values("week")

    # Require at least 4 games
    if len(games) < 4:
        continue

    # Last 3 games
    recent = games.tail(3)

    # All games before the last 3
    baseline = games.iloc[:-3]

    # Calculate averages
    recent_carries = recent["carries"].mean()
    baseline_carries = baseline["carries"].mean()

    recent_yards = recent["rushing_yards"].mean()
    baseline_yards = baseline["rushing_yards"].mean()

    # Calculate YPC using total yards / total carries
    recent_ypc = (
        recent["rushing_yards"].sum() / recent["carries"].sum()
        if recent["carries"].sum() > 0 else 0
    )

    
    # Calculate average team carry share
    recent_share = recent["carry_share"].mean()
    baseline_share = baseline["carry_share"].mean()


    baseline_ypc = (
        baseline["rushing_yards"].sum() / baseline["carries"].sum()
        if baseline["carries"].sum() > 0 else 0
    )

    results.append({
        "player": games["player_name"].iloc[-1],
        "recent_carries": round(recent_carries, 2),
        "baseline_carries": round(baseline_carries, 2),
        "recent_yards": round(recent_yards, 2),
        "baseline_yards": round(baseline_yards, 2),
        "recent_ypc": round(recent_ypc, 2),
        "baseline_ypc": round(baseline_ypc, 2),
        "recent_share": round(recent_share, 2),
        "baseline_share": round(baseline_share, 2)
    })

momentum = pd.DataFrame(results)

#print("\nRUNNING BACK PERFORMANCE COMPARISON")
#print(momentum.to_string(index=False))


# STEP 4: Current Momentum Score

# Calculate percentage change while avoiding division by zero
def percent_change(recent, baseline):
    if baseline <= 0:
        return 0

    change = ((recent - baseline) / baseline) * 100

    # Cap extreme changes at +/- 100%
    return max(-100, min(100, change))


# Calculate changes for each statistic
momentum["carries_change"] = momentum.apply(
    lambda row: percent_change(
        row["recent_carries"], row["baseline_carries"]
    ), axis=1
)

momentum["yards_change"] = momentum.apply(
    lambda row: percent_change(
        row["recent_yards"], row["baseline_yards"]
    ), axis=1
)

momentum["ypc_change"] = momentum.apply(
    lambda row: percent_change(
        row["recent_ypc"], row["baseline_ypc"]
    ), axis=1
)

momentum["share_change"] = momentum.apply(
    lambda row: percent_change(
        row["recent_share"], row["baseline_share"]
    ), axis=1
)


# Calculate weighted momentum score
momentum["momentum_score"] = (
    0.35 * momentum["yards_change"]
    + 0.30 * momentum["carries_change"]
    + 0.15 * momentum["ypc_change"]
    + 0.20 * momentum["share_change"]
)


# Adjust momentum scores based on workload

# Average carries across recent and baseline periods
momentum["workload"] = (
    momentum["recent_carries"] +
    momentum["baseline_carries"]
) / 2

# Full confidence at 12+ carries per game
momentum["reliability"] = (
    momentum["workload"] / 12
).clip(upper=1)

# Adjust momentum score
momentum["momentum_score"] = (
    momentum["momentum_score"] *
    momentum["reliability"]
)


# Classify players
def classify_momentum(score):
    if score >= 20:
        return "HEATING UP"
    elif score >= 5:
        return "TRENDING UP"
    elif score > -5:
        return "STABLE"
    elif score > -20:
        return "COOLING DOWN"
    else:
        return "COLD"


momentum["trend"] = momentum["momentum_score"].apply(
    classify_momentum
)

# Sort from hottest to coldest
momentum = momentum.sort_values(
    "momentum_score", ascending=False
)

#print("\nCURRENT RB MOMENTUM LEADERBOARD")

#print(
#    momentum[[
#        "player",
#        "momentum_score",
#        "trend"
#    ]].round(2).to_string(index=False)
#)


# STEP 5: Future Outlook Score

momentum["future_score"] = (
    0.35 * momentum["carries_change"]
    + 0.35 * momentum["share_change"]
    + 0.15 * momentum["yards_change"]
    + 0.15 * momentum["ypc_change"]
)

# Apply the same workload reliability adjustment
momentum["future_score"] = (
    momentum["future_score"] *
    momentum["reliability"]
)


# Classify future outlook
def classify_future(score):
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


momentum["future_trend"] = momentum["future_score"].apply(
    classify_future
)

# Sort by future outlook
future = momentum.sort_values(
    "future_score", ascending=False
)

#print("\nFUTURE RB OUTLOOK LEADERBOARD")

# print(
#    future[[
#        "player",
#        "momentum_score",
#        "future_score",
#        "future_trend"
#    ]].round(2).to_string(index=False)
#)



# SIMPLE REGRESSION MODEL

momentum["regression_gap"] = (
    momentum["recent_yards"] -
    momentum["baseline_yards"]
)


# How much production changed due to carries
momentum["workload_effect"] = (
    (momentum["recent_carries"] - momentum["baseline_carries"])
    * momentum["baseline_ypc"]
)

# How much production changed due to efficiency
momentum["efficiency_effect"] = (
    momentum["regression_gap"] -
    momentum["workload_effect"]
)

# Identify the biggest factor
momentum["main_driver"] = "EFFICIENCY"

momentum.loc[
    momentum["workload_effect"].abs() >
    momentum["efficiency_effect"].abs(),
    "main_driver"
] = "WORKLOAD"


# Only show players with gaps greater than 20 yards
regression = momentum[
    abs(momentum["regression_gap"]) > 20
].copy()

# Sort from biggest positive gap to biggest negative gap
regression = regression.sort_values(
    "regression_gap", ascending=False
)

# print("\nRB REGRESSION CANDIDATES")


# print(
#    regression[[
#        "player",
#        "regression_gap",
#        "workload_effect",
#        "efficiency_effect",
#        "main_driver"
#    ]].round(2).to_string(index=False)



# FINAL RB REGRESSION LEADERBOARD

table = regression.copy()

# Predict regression direction
table["outlook"] = "BOUNCE BACK"

table.loc[
    table["regression_gap"] > 0,
    "outlook"
] = "REGRESS DOWN"

# Sort by largest absolute regression gap
table["strength"] = table["regression_gap"].abs()

table = table.sort_values(
    "strength", ascending=False
)

# Make the table easy to read
table = table[[
    "player",
    "momentum_score",
    "regression_gap",
    "outlook",
    "main_driver"
]]

table.columns = [
    "PLAYER",
    "MOMENTUM",
    "GAP",
    "OUTLOOK",
    "DRIVER"
]

print("\n========== RB REGRESSION LEADERBOARD ==========\n")

print(table.round(2).to_string(index=False))
