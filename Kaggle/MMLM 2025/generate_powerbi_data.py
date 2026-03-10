"""
generate_powerbi_data.py
------------------------
Reads the March Machine Learning Mania 2025 Kaggle data and writes five CSV files
that are imported directly into the Power BI Team Matchups dashboard:

    powerbi_data/
        matchup_facts.csv       – one row per tourney game (scores, seeds, outcome)
        team_season_stats.csv   – season-average box-score stats per team per season
        team_names.csv          – TeamID → TeamName + gender lookup
        seeds.csv               – Season / TeamID / numeric seed
        predicted_matchups.csv  – every possible 2025 pairing with submitted win-prob
                                  (requires --submission_path)

Usage
-----
    python generate_powerbi_data.py \\
        --data_path  /path/to/march-machine-learning-mania-2025 \\
        --submission_path /path/to/submission.csv

Only requires: pandas, numpy, statsmodels
"""

import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_path",
    default="/kaggle/input/march-machine-learning-mania-2025/",
    help="Folder containing the Kaggle CSV files",
)
parser.add_argument(
    "--out_dir",
    default=os.path.join(os.path.dirname(__file__), "powerbi_data"),
    help="Output folder for the Power BI CSVs",
)
parser.add_argument(
    "--submission_path",
    default=None,
    help="Path to your Kaggle submission CSV (ID, Pred) — enables predicted_matchups.csv",
)
args = parser.parse_args()

DATA_PATH = args.data_path.rstrip("/") + "/"
OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Reading from : {DATA_PATH}")
print(f"Writing to   : {OUT_DIR}")

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
massey = pd.read_csv(DATA_PATH + "MMasseyOrdinals.csv")

tourney_results_raw = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MNCAATourneyDetailedResults.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneyDetailedResults.csv"),
    ],
    ignore_index=True,
)

seeds_raw = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MNCAATourneySeeds.csv"),
        pd.read_csv(DATA_PATH + "WNCAATourneySeeds.csv"),
    ],
    ignore_index=True,
)

regular_results_raw = pd.concat(
    [
        pd.read_csv(DATA_PATH + "MRegularSeasonDetailedResults.csv"),
        pd.read_csv(DATA_PATH + "WRegularSeasonDetailedResults.csv"),
    ],
    ignore_index=True,
)

m_teams = pd.read_csv(DATA_PATH + "MTeams.csv")[["TeamID", "TeamName"]]
m_teams["Gender"] = "M"
w_teams = pd.read_csv(DATA_PATH + "WTeams.csv")[["TeamID", "TeamName"]]
w_teams["Gender"] = "W"
team_names = pd.concat([m_teams, w_teams], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Symmetrise results into T1/T2 framing
# ---------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    dfswap = df[
        [
            "Season", "DayNum", "LTeamID", "LScore", "WTeamID", "WScore",
            "WLoc", "NumOT",
            "LFGM", "LFGA", "LFGM3", "LFGA3", "LFTM", "LFTA",
            "LOR", "LDR", "LAst", "LTO", "LStl", "LBlk", "LPF",
            "WFGM", "WFGA", "WFGM3", "WFGA3", "WFTM", "WFTA",
            "WOR", "WDR", "WAst", "WTO", "WStl", "WBlk", "WPF",
        ]
    ].copy()

    dfswap.loc[df["WLoc"] == "H", "WLoc"] = "A"
    dfswap.loc[df["WLoc"] == "A", "WLoc"] = "H"
    df = df.copy()
    df.columns.values[6] = "location"
    dfswap.columns.values[6] = "location"

    df.columns = [x.replace("W", "T1_").replace("L", "T2_") for x in df.columns]
    dfswap.columns = [x.replace("L", "T1_").replace("W", "T2_") for x in df.columns]

    out = pd.concat([df, dfswap]).reset_index(drop=True)
    out["location"] = out["location"].map({"N": 0, "H": 1, "A": -1}).fillna(0).astype(int)
    out["PointDiff"] = out["T1_Score"] - out["T2_Score"]
    return out


regular_data = prepare_data(regular_results_raw)
tourney_data = prepare_data(tourney_results_raw)


# ---------------------------------------------------------------------------
# 3. Regular-season averages per team per season
# ---------------------------------------------------------------------------
boxscore_cols = [
    "T1_FGM", "T1_FGA", "T1_FGM3", "T1_FGA3",
    "T1_OR", "T1_DR", "T1_Ast", "T1_TO", "T1_Stl", "T1_Blk", "T1_PF",
    "T2_FGM", "T2_FGA", "T2_FGM3", "T2_FGA3",
    "T2_OR", "T2_Ast", "T2_TO", "T2_Stl", "T2_Blk",
    "PointDiff",
]

season_stats = (
    regular_data.groupby(["Season", "T1_TeamID"])[boxscore_cols]
    .mean()
    .reset_index()
)
season_stats.columns = [
    col.replace("T1_", "").replace("T2_", "opp_") if col not in ("Season", "T1_TeamID") else col
    for col in season_stats.columns
]
season_stats.rename(columns={"T1_TeamID": "TeamID"}, inplace=True)

reg_wins = regular_data.copy()
reg_wins["Win"] = (reg_wins["PointDiff"] > 0).astype(int)
win_pct = (
    reg_wins.groupby(["Season", "T1_TeamID"])["Win"]
    .mean()
    .reset_index()
    .rename(columns={"T1_TeamID": "TeamID", "Win": "WinPct"})
)
season_stats = pd.merge(season_stats, win_pct, on=["Season", "TeamID"], how="left")


# ---------------------------------------------------------------------------
# 4. Seeds
# ---------------------------------------------------------------------------
seeds_raw["SeedNum"] = seeds_raw["Seed"].str[1:3].astype(int)
seeds_df = seeds_raw[["Season", "TeamID", "Seed", "SeedNum"]].copy()

seeds_T1 = seeds_df.rename(columns={"TeamID": "T1_TeamID", "SeedNum": "T1_seed"})
seeds_T2 = seeds_df.rename(columns={"TeamID": "T2_TeamID", "SeedNum": "T2_seed"})


# ---------------------------------------------------------------------------
# 5. GLM team quality (Bradley-Terry strength rating, one per season)
#    Fast logistic regression — not a predictive model, just a team-strength
#    metric for the Season Overview page.
# ---------------------------------------------------------------------------
reg_effects = regular_data[["Season", "T1_TeamID", "T2_TeamID", "PointDiff"]].copy()
reg_effects["T1_TeamID"] = reg_effects["T1_TeamID"].astype(str)
reg_effects["T2_TeamID"] = reg_effects["T2_TeamID"].astype(str)
reg_effects["win"] = (reg_effects["PointDiff"] > 0).astype(int)

march_madness = pd.merge(
    seeds_raw[["Season", "TeamID"]], seeds_raw[["Season", "TeamID"]], on="Season"
)
march_madness.columns = ["Season", "T1_TeamID", "T2_TeamID"]
march_madness["T1_TeamID"] = march_madness["T1_TeamID"].astype(str)
march_madness["T2_TeamID"] = march_madness["T2_TeamID"].astype(str)
reg_effects = pd.merge(reg_effects, march_madness, on=["Season", "T1_TeamID", "T2_TeamID"])


def team_quality(season: int) -> pd.DataFrame:
    subset = reg_effects[reg_effects.Season == season]
    if subset.empty:
        return pd.DataFrame(columns=["TeamID", "quality", "Season"])
    try:
        glm = sm.GLM.from_formula(
            "win~-1+T1_TeamID+T2_TeamID",
            data=subset,
            family=sm.families.Binomial(),
        ).fit(disp=False)
        q = pd.DataFrame(glm.params).reset_index()
        q.columns = ["TeamID", "quality"]
        q["Season"] = season
        q = q[q.TeamID.str.contains("T1_")].copy()
        q["TeamID"] = q["TeamID"].str[10:14].astype(int)
    except Exception:
        q = pd.DataFrame(columns=["TeamID", "quality", "Season"])
    return q


print("Computing GLM team quality ratings …")
seasons = sorted(reg_effects.Season.unique())
glm_quality = pd.concat([team_quality(s) for s in seasons], ignore_index=True)
glm_quality_T1 = glm_quality.rename(columns={"TeamID": "T1_TeamID", "quality": "T1_quality"})
glm_quality_T2 = glm_quality.rename(columns={"TeamID": "T2_TeamID", "quality": "T2_quality"})


# ---------------------------------------------------------------------------
# 6. Massey average rankings per team per season
# ---------------------------------------------------------------------------
latest_ranking_days = (
    massey.groupby("Season")["RankingDayNum"]
    .apply(lambda x: x[x < 133].max() if any(x < 133) else x.max())
    .reset_index()
    .rename(columns={"RankingDayNum": "LatestRankingDay"})
)
massey_latest = pd.merge(massey, latest_ranking_days, on="Season")
massey_latest = massey_latest[massey_latest.RankingDayNum == massey_latest.LatestRankingDay]

avg_ranks = (
    massey_latest.groupby(["Season", "TeamID"])["OrdinalRank"]
    .mean()
    .reset_index()
    .rename(columns={"OrdinalRank": "AvgRank"})
)
avg_ranks_T1 = avg_ranks.rename(columns={"TeamID": "T1_TeamID", "AvgRank": "T1_AvgRank"})
avg_ranks_T2 = avg_ranks.rename(columns={"TeamID": "T2_TeamID", "AvgRank": "T2_AvgRank"})


# ---------------------------------------------------------------------------
# 7. Assemble tournament matchup facts
# ---------------------------------------------------------------------------
td = tourney_data[["Season", "DayNum", "T1_TeamID", "T1_Score", "T2_TeamID", "T2_Score"]].copy()
td = td.merge(seeds_T1[["Season", "T1_TeamID", "T1_seed"]], on=["Season", "T1_TeamID"], how="left")
td = td.merge(seeds_T2[["Season", "T2_TeamID", "T2_seed"]], on=["Season", "T2_TeamID"], how="left")
td = td.merge(glm_quality_T1, on=["Season", "T1_TeamID"], how="left")
td = td.merge(glm_quality_T2, on=["Season", "T2_TeamID"], how="left")
td = td.merge(avg_ranks_T1, on=["Season", "T1_TeamID"], how="left")
td = td.merge(avg_ranks_T2, on=["Season", "T2_TeamID"], how="left")

td["Seed_diff"]    = td["T1_seed"] - td["T2_seed"]
td["AvgRank_diff"] = td["T1_AvgRank"] - td["T2_AvgRank"]
td["PointDiff"]    = td["T1_Score"] - td["T2_Score"]
td["IsMensTeam"]   = (td["T1_TeamID"] < 1500).astype(int)
td["Winner"]       = np.where(td["T1_Score"] > td["T2_Score"], "T1", "T2")
td["Upset"]        = np.where(
    (td["Seed_diff"] > 0) & (td["Winner"] == "T1") |
    (td["Seed_diff"] < 0) & (td["Winner"] == "T2"),
    1, 0
)


# ---------------------------------------------------------------------------
# 8. Write CSVs
# ---------------------------------------------------------------------------
print("Writing CSVs …")

# --- matchup_facts.csv ---
matchup_cols = [
    "Season", "DayNum",
    "T1_TeamID", "T1_Score", "T1_seed",
    "T2_TeamID", "T2_Score", "T2_seed",
    "Seed_diff", "PointDiff", "Winner", "Upset",
    "T1_quality", "T2_quality", "T1_AvgRank", "T2_AvgRank", "AvgRank_diff",
    "IsMensTeam",
]
matchup_facts = td[[c for c in matchup_cols if c in td.columns]].copy()
matchup_facts = matchup_facts.merge(
    team_names.rename(columns={"TeamID": "T1_TeamID", "TeamName": "T1_TeamName", "Gender": "T1_Gender"}),
    on="T1_TeamID", how="left",
)
matchup_facts = matchup_facts.merge(
    team_names.rename(columns={"TeamID": "T2_TeamID", "TeamName": "T2_TeamName", "Gender": "T2_Gender"}),
    on="T2_TeamID", how="left",
)
matchup_facts.to_csv(os.path.join(OUT_DIR, "matchup_facts.csv"), index=False)
print(f"  matchup_facts.csv        : {len(matchup_facts):,} rows")

# --- team_season_stats.csv ---
team_season_stats = season_stats.merge(
    glm_quality[["TeamID", "Season", "quality"]], on=["TeamID", "Season"], how="left"
)
team_season_stats = team_season_stats.merge(avg_ranks, on=["TeamID", "Season"], how="left")
team_season_stats = team_season_stats.merge(team_names, on="TeamID", how="left")
team_season_stats.to_csv(os.path.join(OUT_DIR, "team_season_stats.csv"), index=False)
print(f"  team_season_stats.csv    : {len(team_season_stats):,} rows")

# --- team_names.csv ---
team_names.to_csv(os.path.join(OUT_DIR, "team_names.csv"), index=False)
print(f"  team_names.csv           : {len(team_names):,} rows")

# --- seeds.csv ---
seeds_raw.merge(team_names, on="TeamID", how="left").to_csv(
    os.path.join(OUT_DIR, "seeds.csv"), index=False
)
print(f"  seeds.csv                : {len(seeds_raw):,} rows")

# --- predicted_matchups.csv (from submission file) ---
if args.submission_path:
    sub = pd.read_csv(args.submission_path)
    sub[["Season", "T1_TeamID", "T2_TeamID"]] = (
        sub["ID"].str.split("_", expand=True).iloc[:, :3].astype(int).values
    )
    sub = sub.rename(columns={"Pred": "T1_WinProb"})
    sub["T2_WinProb"] = 1 - sub["T1_WinProb"]

    sub = sub.merge(
        team_names.rename(columns={"TeamID": "T1_TeamID", "TeamName": "T1_TeamName", "Gender": "T1_Gender"}),
        on="T1_TeamID", how="left",
    )
    sub = sub.merge(
        team_names.rename(columns={"TeamID": "T2_TeamID", "TeamName": "T2_TeamName", "Gender": "T2_Gender"}),
        on="T2_TeamID", how="left",
    )

    seeds_2025 = seeds_df[seeds_df.Season == 2025][["TeamID", "SeedNum", "Seed"]].copy()
    sub = sub.merge(
        seeds_2025.rename(columns={"TeamID": "T1_TeamID", "SeedNum": "T1_seed", "Seed": "T1_Seed"}),
        on="T1_TeamID", how="left",
    )
    sub = sub.merge(
        seeds_2025.rename(columns={"TeamID": "T2_TeamID", "SeedNum": "T2_seed", "Seed": "T2_Seed"}),
        on="T2_TeamID", how="left",
    )

    sub["Seed_diff"]        = sub["T1_seed"] - sub["T2_seed"]
    sub["Gender"]           = sub["T1_Gender"]
    sub["FavouriteWinProb"] = sub[["T1_WinProb", "T2_WinProb"]].max(axis=1)
    sub["Favourite"]        = np.where(sub["T1_WinProb"] >= 0.5, sub["T1_TeamName"], sub["T2_TeamName"])
    sub["Underdog"]         = np.where(sub["T1_WinProb"] >= 0.5, sub["T2_TeamName"], sub["T1_TeamName"])
    sub["MatchupLabel"]     = sub["T1_TeamName"] + " vs " + sub["T2_TeamName"]
    sub["Upset_prob"]       = np.where(
        sub["T1_seed"] < sub["T2_seed"], sub["T2_WinProb"],
        np.where(sub["T1_seed"] > sub["T2_seed"], sub["T1_WinProb"], np.nan)
    )

    out_cols = [
        "ID", "Season",
        "T1_TeamID", "T1_TeamName", "T1_Gender", "T1_Seed", "T1_seed", "T1_WinProb",
        "T2_TeamID", "T2_TeamName", "T2_Gender", "T2_Seed", "T2_seed", "T2_WinProb",
        "Seed_diff", "Favourite", "Underdog", "FavouriteWinProb",
        "MatchupLabel", "Upset_prob", "Gender",
    ]
    sub[[c for c in out_cols if c in sub.columns]].to_csv(
        os.path.join(OUT_DIR, "predicted_matchups.csv"), index=False
    )
    print(f"  predicted_matchups.csv   : {len(sub):,} rows  ({sub['T1_TeamID'].nunique()} teams × all pairings)")
else:
    print("  predicted_matchups.csv   : skipped (pass --submission_path to include)")

print("\nDone. Import the CSV files into Power BI using the template.")
