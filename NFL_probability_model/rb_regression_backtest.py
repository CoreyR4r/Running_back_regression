
import nflreadpy as nfl
import pandas as pd

stats = nfl.load_player_stats([2024, 2025]).to_pandas()
stats = stats[stats["season_type"] == "REG"]

rb = stats[stats["position"] == "RB"].copy()

results = []

for (season, player), games in rb.groupby(
    ["season", "player_id"]
):

    games = games.sort_values("week").reset_index(drop=True)

    if len(games) < 7:
        continue

    for i in range(4, len(games) - 2, 3):

        past = games.iloc[:i]
        recent = past.tail(3)
        baseline = past.iloc[:-3]
        future = games.iloc[i:i+3]

        recent_carries = recent["carries"].mean()
        baseline_carries = baseline["carries"].mean()

        recent_yards = recent["rushing_yards"].mean()

        baseline_attempts = baseline["carries"].sum()

        if baseline_attempts == 0:
            continue

        baseline_ypc = (
            baseline["rushing_yards"].sum()
            / baseline_attempts
        )

        expected_carries = (
            recent_carries + baseline_carries
        ) / 2

        expected_yards = expected_carries * baseline_ypc

        actual_yards = future["rushing_yards"].mean()

        
        regression_gap = recent_yards - expected_yards

        
    # Separate workload and efficiency

        workload_effect = (
            (recent_carries - baseline_carries)
            * baseline_ypc
        )

        efficiency_effect = (
            regression_gap - workload_effect
        )

        if abs(workload_effect) > abs(efficiency_effect):
            main_driver = "WORKLOAD"
        else:
            main_driver = "EFFICIENCY"


        if regression_gap > 0:
            correct = actual_yards < recent_yards
        elif regression_gap < 0:
            correct = actual_yards > recent_yards
        else:
            correct = False

        
        # Simple baseline: predict regression toward earlier average
        baseline_yards = baseline["rushing_yards"].mean()

        if recent_yards > baseline_yards:
            baseline_correct = actual_yards < recent_yards
        elif recent_yards < baseline_yards:
            baseline_correct = actual_yards > recent_yards
        else:
            baseline_correct = False


        results.append({
            "season": season,
            "expected_error": abs(expected_yards - actual_yards),
            "recent_error": abs(recent_yards - actual_yards),
            "regression_gap": regression_gap,
            "correct_direction": correct,
            "baseline_correct": baseline_correct,
            "baseline_gap": recent_yards - baseline_yards,
            "main_driver": main_driver,
        })



backtest = pd.DataFrame(results)

print("\nREGRESSION MODEL BACKTEST")
print("Predictions tested:", len(backtest))

print(
    "Regression Model MAE:",
    round(backtest["expected_error"].mean(), 2)
)

print(
    "Recent 3-Game Average MAE:",
    round(backtest["recent_error"].mean(), 2)
)

print("\nRESULTS BY SEASON")

print(
    backtest.groupby("season")[
        ["expected_error", "recent_error"]
    ].mean().round(2).to_string()
)


print("\nREGRESSION DIRECTION ACCURACY")

for minimum_gap in [0, 10, 20, 30]:
    test = backtest[
        abs(backtest["regression_gap"]) > minimum_gap
    ]

    accuracy = test["correct_direction"].mean() * 100

    print(
        "Gap greater than", minimum_gap,
        "| Predictions:", len(test),
        "| Accuracy:", round(accuracy, 2), "%"
    )


print("\nMODEL VS SIMPLE BASELINE")

for minimum_gap in [0, 10, 20, 30]:
    test = backtest[
        abs(backtest["regression_gap"]) > minimum_gap
    ]

    model_accuracy = test["correct_direction"].mean() * 100
    baseline_accuracy = test["baseline_correct"].mean() * 100

    print("\nGap greater than", minimum_gap)
    print("Predictions:", len(test))
    print("Model accuracy:", round(model_accuracy, 2), "%")
    print("Baseline accuracy:", round(baseline_accuracy, 2), "%")


print("\nSIMPLE REGRESSION MODEL")

for minimum_gap in [0, 10, 20, 30]:
    test = backtest[
        abs(backtest["baseline_gap"]) > minimum_gap
    ]

    accuracy = test["baseline_correct"].mean() * 100

    print(
        "Gap greater than", minimum_gap,
        "| Predictions:", len(test),
        "| Accuracy:", round(accuracy, 2), "%"
    )

print("\nACCURACY BY SEASON")

for season in [2024, 2025]:
    test = backtest[
        (backtest["season"] == season) &
        (abs(backtest["baseline_gap"]) > 20)
    ]

    accuracy = test["baseline_correct"].mean() * 100

    print(
        season,
        "| Predictions:", len(test),
        "| Accuracy:", round(accuracy, 2), "%"
    )


print("\nREGRESSION ACCURACY BY DRIVER")

test = backtest[
    abs(backtest["baseline_gap"]) > 20
]

summary = test.groupby("main_driver").agg(
    predictions=("baseline_correct", "count"),
    accuracy=("baseline_correct", "mean")
)

summary["accuracy"] *= 100

print(summary.round(2).to_string())

print("\nACCURACY BY DRIVER AND SEASON")

summary = test.groupby(
    ["season", "main_driver"]
).agg(
    predictions=("baseline_correct", "count"),
    accuracy=("baseline_correct", "mean")
)

summary["accuracy"] *= 100

print(summary.round(2).to_string())


print("\nUPWARD VS DOWNWARD REGRESSION")

test = backtest[
    abs(backtest["baseline_gap"]) > 20
].copy()

test["direction"] = "UPWARD"

test.loc[
    test["baseline_gap"] > 0,
    "direction"
] = "DOWNWARD"

summary = test.groupby("direction").agg(
    predictions=("baseline_correct", "count"),
    accuracy=("baseline_correct", "mean")
)

summary["accuracy"] *= 100

print(summary.round(2).to_string())
