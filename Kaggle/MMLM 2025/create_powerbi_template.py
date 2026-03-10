"""
create_powerbi_template.py
--------------------------
Generates  TeamMatchupsDashboard.pbit  – a Power BI template file that
contains:
  • Data model (5 tables, relationships, DAX measures)
  • Report layout (3 pages):
      1. Team Matchup      – historical games filtered by season / team
      2. Season Overview   – upset rates, seed win %, quality scatter
      3. Matchup Explorer  – pick any two 2025 teams, see predicted win
                             probability from the submitted model

Run AFTER generate_powerbi_data.py so the powerbi_data/ CSVs exist,
then open TeamMatchupsDashboard.pbit in Power BI Desktop.

Power BI will prompt you to set the data-source path to the folder that
contains the five CSV files.
"""

import json
import os
import zipfile

OUT_DIR = os.path.join(os.path.dirname(__file__), "powerbi_data")
PBIT_PATH = os.path.join(os.path.dirname(__file__), "TeamMatchupsDashboard.pbit")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col(name, dtype="string", hidden=False):
    out = {"Name": name, "DataType": dtype, "IsHidden": hidden}
    return out


def _measure(name, expr, fmt="", hidden=False):
    m = {"Name": name, "Expression": expr, "IsHidden": hidden}
    if fmt:
        m["FormatString"] = fmt
    return m


# ---------------------------------------------------------------------------
# Data-model schema
# ---------------------------------------------------------------------------
DATA_MODEL = {
    "name": "TeamMatchupsDashboard",
    "compatibilityLevel": 1550,
    "model": {
        "defaultPowerBIDataSourceVersion": "powerBI_V3",
        "tables": [
            # -------- matchup_facts ----------------------------------------
            {
                "name": "matchup_facts",
                "columns": [
                    _col("Season",        "int64"),
                    _col("DayNum",        "int64"),
                    _col("T1_TeamID",     "int64"),
                    _col("T1_TeamName",   "string"),
                    _col("T1_Score",      "double"),
                    _col("T1_seed",       "double"),
                    _col("T1_WinProb",    "double"),
                    _col("T2_TeamID",     "int64"),
                    _col("T2_TeamName",   "string"),
                    _col("T2_Score",      "double"),
                    _col("T2_seed",       "double"),
                    _col("T2_WinProb",    "double"),
                    _col("Seed_diff",     "double"),
                    _col("Winner",        "string"),
                    _col("T1_quality",    "double"),
                    _col("T2_quality",    "double"),
                    _col("T1_AvgRank",    "double"),
                    _col("T2_AvgRank",    "double"),
                    _col("AvgRank_diff",  "double"),
                    _col("IsMensTeam",    "int64"),
                ],
                "measures": [
                    _measure(
                        "T1 Win %",
                        "AVERAGE(matchup_facts[T1_WinProb])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "T2 Win %",
                        "AVERAGE(matchup_facts[T2_WinProb])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Avg Point Margin",
                        "AVERAGEX(matchup_facts, matchup_facts[T1_Score] - matchup_facts[T2_Score])",
                        fmt="+0.0;-0.0;0.0",
                    ),
                    _measure(
                        "Total Games",
                        "COUNTROWS(matchup_facts)",
                        fmt="#,0",
                    ),
                    _measure(
                        "T1 Actual Wins",
                        'COUNTROWS(FILTER(matchup_facts, matchup_facts[Winner] = "T1"))',
                        fmt="#,0",
                    ),
                    _measure(
                        "T1 Win Rate",
                        'DIVIDE([T1 Actual Wins], [Total Games])',
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Seed Upset",
                        # 1 when a higher-seeded (worse) team wins
                        "COUNTROWS(FILTER(matchup_facts, matchup_facts[Seed_diff] > 0 && matchup_facts[Winner] = \"T1\")) + "
                        "COUNTROWS(FILTER(matchup_facts, matchup_facts[Seed_diff] < 0 && matchup_facts[Winner] = \"T2\"))",
                        fmt="#,0",
                    ),
                    _measure(
                        "Upset Rate",
                        "DIVIDE([Seed Upset], [Total Games])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Quality Diff",
                        "AVERAGE(matchup_facts[T1_quality]) - AVERAGE(matchup_facts[T2_quality])",
                        fmt="+0.00;-0.00;0.00",
                    ),
                ],
                "partitions": [
                    {
                        "name": "matchup_facts",
                        "source": {
                            "type": "m",
                            "expression": [
                                "let",
                                '    Source = Csv.Document(File.Contents(_DataFolder & "matchup_facts.csv"),[Delimiter=",", Columns=20, Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                                "    #\"Changed Types\" = Table.TransformColumnTypes(#\"Promoted Headers\",{",
                                '        {"Season", Int64.Type}, {"DayNum", Int64.Type},',
                                '        {"T1_TeamID", Int64.Type}, {"T2_TeamID", Int64.Type},',
                                '        {"T1_Score", type number}, {"T2_Score", type number},',
                                '        {"T1_seed", type number}, {"T2_seed", type number},',
                                '        {"T1_WinProb", type number}, {"T2_WinProb", type number},',
                                '        {"Seed_diff", type number}, {"T1_quality", type number},',
                                '        {"T2_quality", type number}, {"T1_AvgRank", type number},',
                                '        {"T2_AvgRank", type number}, {"AvgRank_diff", type number},',
                                '        {"IsMensTeam", Int64.Type}',
                                "    })",
                                "in",
                                "    #\"Changed Types\"",
                            ],
                        },
                    }
                ],
            },
            # -------- team_season_stats ------------------------------------
            {
                "name": "team_season_stats",
                "columns": [
                    _col("Season",      "int64"),
                    _col("TeamID",      "int64"),
                    _col("TeamName",    "string"),
                    _col("Gender",      "string"),
                    _col("FGM",         "double"),
                    _col("FGA",         "double"),
                    _col("FGM3",        "double"),
                    _col("FGA3",        "double"),
                    _col("OR",          "double"),
                    _col("DR",          "double"),
                    _col("Ast",         "double"),
                    _col("TO",          "double"),
                    _col("Stl",         "double"),
                    _col("Blk",         "double"),
                    _col("PF",          "double"),
                    _col("opp_FGM",     "double"),
                    _col("opp_FGA",     "double"),
                    _col("opp_FGM3",    "double"),
                    _col("opp_FGA3",    "double"),
                    _col("opp_OR",      "double"),
                    _col("opp_Ast",     "double"),
                    _col("opp_TO",      "double"),
                    _col("opp_Stl",     "double"),
                    _col("opp_Blk",     "double"),
                    _col("PointDiff",   "double"),
                    _col("WinPct",      "double"),
                    _col("quality",     "double"),
                    _col("AvgRank",     "double"),
                ],
                "measures": [
                    _measure("Avg FG%",   "DIVIDE(SUM(team_season_stats[FGM]), SUM(team_season_stats[FGA]))", "#0.0%"),
                    _measure("Avg 3P%",   "DIVIDE(SUM(team_season_stats[FGM3]), SUM(team_season_stats[FGA3]))", "#0.0%"),
                    _measure("Avg PPG",   "AVERAGEX(team_season_stats, team_season_stats[FGM]*2 + team_season_stats[FGM3] + team_season_stats[PointDiff])", "0.0"),
                    _measure("Avg Assists","AVERAGE(team_season_stats[Ast])", "0.0"),
                    _measure("Avg Turnovers","AVERAGE(team_season_stats[TO])", "0.0"),
                    _measure("Ast/TO Ratio","DIVIDE([Avg Assists],[Avg Turnovers])", "0.00"),
                    _measure("Avg Win %", "AVERAGE(team_season_stats[WinPct])", "#0.0%"),
                ],
                "partitions": [
                    {
                        "name": "team_season_stats",
                        "source": {
                            "type": "m",
                            "expression": [
                                "let",
                                '    Source = Csv.Document(File.Contents(_DataFolder & "team_season_stats.csv"),[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                                "    #\"Changed Types\" = Table.TransformColumnTypes(#\"Promoted Headers\",{",
                                '        {"Season", Int64.Type}, {"TeamID", Int64.Type},',
                                '        {"FGM", type number}, {"FGA", type number},',
                                '        {"FGM3", type number}, {"FGA3", type number},',
                                '        {"OR", type number}, {"DR", type number},',
                                '        {"Ast", type number}, {"TO", type number},',
                                '        {"Stl", type number}, {"Blk", type number},',
                                '        {"PF", type number}, {"opp_FGM", type number},',
                                '        {"opp_FGA", type number}, {"PointDiff", type number},',
                                '        {"WinPct", type number}, {"quality", type number},',
                                '        {"AvgRank", type number}',
                                "    })",
                                "in",
                                "    #\"Changed Types\"",
                            ],
                        },
                    }
                ],
            },
            # -------- team_names -------------------------------------------
            {
                "name": "team_names",
                "columns": [
                    _col("TeamID",   "int64"),
                    _col("TeamName", "string"),
                    _col("Gender",   "string"),
                ],
                "partitions": [
                    {
                        "name": "team_names",
                        "source": {
                            "type": "m",
                            "expression": [
                                "let",
                                '    Source = Csv.Document(File.Contents(_DataFolder & "team_names.csv"),[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                                "    #\"Changed Types\" = Table.TransformColumnTypes(#\"Promoted Headers\",{",
                                '        {"TeamID", Int64.Type}',
                                "    })",
                                "in",
                                "    #\"Changed Types\"",
                            ],
                        },
                    }
                ],
            },
            # -------- seeds ------------------------------------------------
            {
                "name": "seeds",
                "columns": [
                    _col("Season",   "int64"),
                    _col("TeamID",   "int64"),
                    _col("Seed",     "string"),
                    _col("SeedNum",  "int64"),
                    _col("TeamName", "string"),
                    _col("Gender",   "string"),
                ],
                "partitions": [
                    {
                        "name": "seeds",
                        "source": {
                            "type": "m",
                            "expression": [
                                "let",
                                '    Source = Csv.Document(File.Contents(_DataFolder & "seeds.csv"),[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                                "    #\"Changed Types\" = Table.TransformColumnTypes(#\"Promoted Headers\",{",
                                '        {"Season", Int64.Type}, {"TeamID", Int64.Type}, {"SeedNum", Int64.Type}',
                                "    })",
                                "in",
                                "    #\"Changed Types\"",
                            ],
                        },
                    }
                ],
            },
            # -------- predicted_matchups (submission CSV – all pairings) ----
            {
                "name": "predicted_matchups",
                "columns": [
                    _col("ID",               "string"),
                    _col("Season",           "int64"),
                    _col("T1_TeamID",        "int64"),
                    _col("T1_TeamName",      "string"),
                    _col("T1_Gender",        "string"),
                    _col("T1_Seed",          "string"),
                    _col("T1_seed",          "double"),
                    _col("T1_WinProb",       "double"),
                    _col("T2_TeamID",        "int64"),
                    _col("T2_TeamName",      "string"),
                    _col("T2_Gender",        "string"),
                    _col("T2_Seed",          "string"),
                    _col("T2_seed",          "double"),
                    _col("T2_WinProb",       "double"),
                    _col("Seed_diff",        "double"),
                    _col("Favourite",        "string"),
                    _col("Underdog",         "string"),
                    _col("FavouriteWinProb", "double"),
                    _col("MatchupLabel",     "string"),
                    _col("Upset_prob",       "double"),
                    _col("Gender",           "string"),
                ],
                "measures": [
                    _measure(
                        "Selected T1 Win Prob",
                        "AVERAGE(predicted_matchups[T1_WinProb])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Selected T2 Win Prob",
                        "AVERAGE(predicted_matchups[T2_WinProb])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Avg Upset Probability",
                        "AVERAGE(predicted_matchups[Upset_prob])",
                        fmt="#0.0%",
                    ),
                    _measure(
                        "Matchup Count",
                        "COUNTROWS(predicted_matchups)",
                        fmt="#,0",
                    ),
                    _measure(
                        "Favourite Avg Win Prob",
                        "AVERAGE(predicted_matchups[FavouriteWinProb])",
                        fmt="#0.0%",
                    ),
                ],
                "partitions": [
                    {
                        "name": "predicted_matchups",
                        "source": {
                            "type": "m",
                            "expression": [
                                "let",
                                '    Source = Csv.Document(File.Contents(_DataFolder & "predicted_matchups.csv"),[Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.None]),',
                                "    #\"Promoted Headers\" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
                                "    #\"Changed Types\" = Table.TransformColumnTypes(#\"Promoted Headers\",{",
                                '        {"Season", Int64.Type}, {"T1_TeamID", Int64.Type}, {"T2_TeamID", Int64.Type},',
                                '        {"T1_seed", type number}, {"T2_seed", type number},',
                                '        {"T1_WinProb", type number}, {"T2_WinProb", type number},',
                                '        {"Seed_diff", type number}, {"FavouriteWinProb", type number},',
                                '        {"Upset_prob", type number}',
                                "    })",
                                "in",
                                "    #\"Changed Types\"",
                            ],
                        },
                    }
                ],
            },
            # -------- _Parameters (data folder path) -----------------------
            {
                "name": "_Parameters",
                "columns": [_col("_DataFolder", "string")],
                "partitions": [
                    {
                        "name": "_Parameters",
                        "source": {
                            "type": "m",
                            "expression": [
                                'let',
                                '    Source = #table(type table [_DataFolder = text], {{_DataFolder}})',
                                'in',
                                '    Source',
                            ],
                        },
                    }
                ],
                "isHidden": True,
            },
        ],
        # -------- Relationships -------------------------------------------
        "relationships": [
            {
                "name": "matchup_T1_team_names",
                "fromTable": "matchup_facts",
                "fromColumn": "T1_TeamID",
                "toTable": "team_names",
                "toColumn": "TeamID",
                "crossFilteringBehavior": "oneDirection",
            },
            {
                "name": "matchup_T1_team_season_stats",
                "fromTable": "matchup_facts",
                "fromColumn": "T1_TeamID",
                "toTable": "team_season_stats",
                "toColumn": "TeamID",
                "crossFilteringBehavior": "oneDirection",
            },
            {
                "name": "matchup_T1_seeds",
                "fromTable": "matchup_facts",
                "fromColumn": "T1_TeamID",
                "toTable": "seeds",
                "toColumn": "TeamID",
                "crossFilteringBehavior": "oneDirection",
            },
            {
                "name": "predicted_T1_team_names",
                "fromTable": "predicted_matchups",
                "fromColumn": "T1_TeamID",
                "toTable": "team_names",
                "toColumn": "TeamID",
                "crossFilteringBehavior": "oneDirection",
            },
        ],
        # -------- Query Groups (Power Query parameters) -------------------
        "queryGroups": [],
        "expressions": [
            {
                "name": "_DataFolder",
                "kind": "m",
                "expression": [
                    '// Set this to the absolute path of your powerbi_data folder',
                    '// e.g.  "C:/Users/you/Data-Science/Kaggle/MMLM 2025/powerbi_data/"',
                    '"C:/powerbi_data/"  meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]',
                ],
            }
        ],
    },
}

# ---------------------------------------------------------------------------
# Report layout – two pages
# ---------------------------------------------------------------------------
# Visual config helpers

def _slicer(x, y, w, h, table, col, display_name, orientation="vertical", vis_id=None):
    vid = vis_id or f"slicer_{col}"
    return {
        "id": vid,
        "type": "slicer",
        "x": x, "y": y, "z": 0,
        "width": w, "height": h,
        "visual": {
            "visualType": "slicer",
            "projections": {
                "Values": [{"queryRef": f"{table}.{col}", "active": True}]
            },
            "prototypeQuery": {
                "Select": [{"Column": {"Expression": {"SourceRef": {"Entity": table}}, "Property": col}, "Name": f"{table}.{col}"}],
                "From": [{"Name": "t", "Entity": table, "Type": 0}],
            },
            "vcObjects": {
                "data": [{"properties": {"mode": {"expr": {"Literal": {"Value": f"'{orientation}'"}}}}}],
                "header": [{"properties": {"text": {"expr": {"Literal": {"Value": f"'{display_name}'"}}}, "show": {"expr": {"Literal": {"Value": "'True'"}}}}}],
            },
        },
    }


def _card(x, y, w, h, measure_table, measure_name, title):
    return {
        "id": f"card_{measure_name.replace(' ', '_')}",
        "type": "card",
        "x": x, "y": y, "z": 0,
        "width": w, "height": h,
        "visual": {
            "visualType": "card",
            "projections": {
                "Values": [{"queryRef": f"{measure_table}.{measure_name}", "active": True}]
            },
            "prototypeQuery": {
                "Select": [{"Measure": {"Expression": {"SourceRef": {"Entity": measure_table}}, "Property": measure_name}, "Name": f"{measure_table}.{measure_name}"}],
                "From": [{"Name": "t", "Entity": measure_table, "Type": 0}],
            },
            "vcObjects": {
                "title": [{"properties": {"text": {"expr": {"Literal": {"Value": f"'{title}'"}}}, "show": {"expr": {"Literal": {"Value": "'True'"}}}}}],
            },
        },
    }


def _clustered_bar(x, y, w, h, vis_id, title, cat_table, cat_col, val_table, val_measures):
    selects = []
    from_clause = [
        {"Name": "c", "Entity": cat_table, "Type": 0},
    ]
    if val_table != cat_table:
        from_clause.append({"Name": "v", "Entity": val_table, "Type": 0})

    selects.append({
        "Column": {"Expression": {"SourceRef": {"Entity": cat_table}}, "Property": cat_col},
        "Name": f"{cat_table}.{cat_col}",
    })
    for m in val_measures:
        selects.append({
            "Measure": {"Expression": {"SourceRef": {"Entity": val_table}}, "Property": m},
            "Name": f"{val_table}.{m}",
        })

    return {
        "id": vis_id,
        "type": "clusteredBarChart",
        "x": x, "y": y, "z": 0,
        "width": w, "height": h,
        "visual": {
            "visualType": "clusteredBarChart",
            "projections": {
                "Category": [{"queryRef": f"{cat_table}.{cat_col}", "active": True}],
                "Y": [{"queryRef": f"{val_table}.{m}", "active": True} for m in val_measures],
            },
            "prototypeQuery": {"Select": selects, "From": from_clause},
            "vcObjects": {
                "title": [{"properties": {
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                    "show": {"expr": {"Literal": {"Value": "'True'"}}}
                }}]
            },
        },
    }


def _scatter(x, y, w, h, vis_id, title, x_table, x_col, y_table, y_col, detail_table, detail_col):
    return {
        "id": vis_id,
        "type": "scatterChart",
        "x": x, "y": y, "z": 0,
        "width": w, "height": h,
        "visual": {
            "visualType": "scatterChart",
            "projections": {
                "X": [{"queryRef": f"{x_table}.{x_col}", "active": True}],
                "Y": [{"queryRef": f"{y_table}.{y_col}", "active": True}],
                "Details": [{"queryRef": f"{detail_table}.{detail_col}", "active": True}],
            },
            "prototypeQuery": {
                "Select": [
                    {"Column": {"Expression": {"SourceRef": {"Entity": x_table}}, "Property": x_col}, "Name": f"{x_table}.{x_col}"},
                    {"Measure": {"Expression": {"SourceRef": {"Entity": y_table}}, "Property": y_col}, "Name": f"{y_table}.{y_col}"},
                    {"Column": {"Expression": {"SourceRef": {"Entity": detail_table}}, "Property": detail_col}, "Name": f"{detail_table}.{detail_col}"},
                ],
                "From": [
                    {"Name": "t", "Entity": x_table, "Type": 0},
                    {"Name": "s", "Entity": detail_table, "Type": 0},
                ],
            },
            "vcObjects": {
                "title": [{"properties": {
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                    "show": {"expr": {"Literal": {"Value": "'True'"}}}
                }}]
            },
        },
    }


def _table_visual(x, y, w, h, vis_id, title, columns):
    """columns: list of (table, col_or_measure, is_measure)"""
    selects = []
    projections = []
    from_seen = {}
    from_clause = []
    for tbl, prop, is_meas in columns:
        if tbl not in from_seen:
            from_seen[tbl] = len(from_seen)
            from_clause.append({"Name": f"t{from_seen[tbl]}", "Entity": tbl, "Type": 0})
        ref = f"{tbl}.{prop}"
        if is_meas:
            selects.append({"Measure": {"Expression": {"SourceRef": {"Entity": tbl}}, "Property": prop}, "Name": ref})
        else:
            selects.append({"Column": {"Expression": {"SourceRef": {"Entity": tbl}}, "Property": prop}, "Name": ref})
        projections.append({"queryRef": ref, "active": True})

    return {
        "id": vis_id,
        "type": "tableEx",
        "x": x, "y": y, "z": 0,
        "width": w, "height": h,
        "visual": {
            "visualType": "tableEx",
            "projections": {"Values": projections},
            "prototypeQuery": {"Select": selects, "From": from_clause},
            "vcObjects": {
                "title": [{"properties": {
                    "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                    "show": {"expr": {"Literal": {"Value": "'True'"}}}
                }}]
            },
        },
    }


# ---- Page 1: Team Matchup ------------------------------------------------
CANVAS_W = 1280
CANVAS_H = 720

page1_visuals = [
    # Slicers (left rail)
    _slicer(10, 10,  200, 60,  "matchup_facts", "Season",      "Season",  "horizontal", "slicer_Season"),
    _slicer(10, 80,  200, 280, "matchup_facts", "T1_TeamName", "Team 1",  "vertical",   "slicer_T1"),
    _slicer(10, 370, 200, 280, "matchup_facts", "T2_TeamName", "Team 2",  "vertical",   "slicer_T2"),
    # KPI cards (top row)
    _card(220, 10, 200, 80, "matchup_facts", "T1 Win %",       "Team 1 Win Probability"),
    _card(430, 10, 200, 80, "matchup_facts", "T2 Win %",       "Team 2 Win Probability"),
    _card(640, 10, 200, 80, "matchup_facts", "Avg Point Margin","Avg Point Margin (T1−T2)"),
    _card(850, 10, 200, 80, "matchup_facts", "Total Games",    "Total Tournament Games"),
    _card(1060,10, 200, 80, "matchup_facts", "Upset Rate",     "Upset Rate"),
    # Stats comparison bar chart
    _clustered_bar(
        220, 100, 640, 280,
        "bar_stats", "Season-Avg Box-Score Comparison",
        "team_season_stats", "TeamName",
        "team_season_stats", ["Avg FG%", "Avg 3P%", "Avg Assists", "Avg Turnovers"],
    ),
    # Win probability scatter: seed vs quality
    _scatter(870, 100, 390, 280,
        "scatter_seed_quality",
        "Seed vs GLM Quality (bubble = win prob)",
        "matchup_facts", "T1_seed",
        "matchup_facts", "T1 Win %",
        "matchup_facts", "T1_TeamName",
    ),
    # Historical matchups table
    _table_visual(
        220, 390, 1040, 300,
        "tbl_matchups",
        "Historical Matchups (filtered by slicers)",
        [
            ("matchup_facts", "Season",      False),
            ("matchup_facts", "DayNum",      False),
            ("matchup_facts", "T1_TeamName", False),
            ("matchup_facts", "T1_Score",    False),
            ("matchup_facts", "T2_TeamName", False),
            ("matchup_facts", "T2_Score",    False),
            ("matchup_facts", "T1_seed",     False),
            ("matchup_facts", "T2_seed",     False),
            ("matchup_facts", "T1_WinProb",  False),
            ("matchup_facts", "Winner",      False),
        ],
    ),
]

# ---- Page 2: Season Overview ---------------------------------------------
page2_visuals = [
    _slicer(10, 10, 200, 60,  "matchup_facts", "Season",  "Season", "horizontal", "slicer_Season2"),
    _slicer(10, 80, 200, 140, "seeds",         "Gender",  "Gender", "vertical",   "slicer_Gender"),
    # Upset rate by season
    _clustered_bar(
        220, 10, 500, 260,
        "bar_upset_season", "Upset Rate by Season",
        "matchup_facts", "Season",
        "matchup_facts", ["Upset Rate"],
    ),
    # Win % by seed
    _clustered_bar(
        730, 10, 530, 260,
        "bar_win_seed", "T1 Win % by T1 Seed Number",
        "matchup_facts", "T1_seed",
        "matchup_facts", ["T1 Win %"],
    ),
    # Quality comparison scatter
    _scatter(220, 280, 530, 260,
        "scatter_quality", "GLM Quality: T1 vs T2",
        "matchup_facts", "T1_quality",
        "matchup_facts", "T2 Win %",
        "matchup_facts", "Season",
    ),
    # Avg Win% by team (top performers)
    _clustered_bar(
        760, 280, 500, 260,
        "bar_team_winpct", "Regular-Season Win % by Team",
        "team_season_stats", "TeamName",
        "team_season_stats", ["Avg Win %"],
    ),
    # Ranking trend table
    _table_visual(
        220, 550, 1040, 160,
        "tbl_rankings",
        "Team Rankings & Stats",
        [
            ("team_season_stats", "Season",    False),
            ("team_season_stats", "TeamName",  False),
            ("team_season_stats", "Gender",    False),
            ("team_season_stats", "AvgRank",   False),
            ("team_season_stats", "quality",   False),
            ("team_season_stats", "WinPct",    False),
            ("team_season_stats", "Avg FG%",   True),
            ("team_season_stats", "Avg 3P%",   True),
            ("team_season_stats", "Ast/TO Ratio", True),
        ],
    ),
]

# ---- Page 3: Matchup Explorer (predicted_matchups) -----------------------
# The submission CSV has every possible 2025 pairing, so slicers here let
# the user pick any Team 1 + Team 2 and see the model's predicted probability.
page3_visuals = [
    # Slicers
    _slicer(10,  10, 220, 50,  "predicted_matchups", "T1_Gender",   "Gender",  "horizontal", "slicer_Gender3"),
    _slicer(10,  70, 220, 280, "predicted_matchups", "T1_TeamName", "Team 1",  "vertical",   "slicer_T1_pred"),
    _slicer(10, 360, 220, 280, "predicted_matchups", "T2_TeamName", "Team 2",  "vertical",   "slicer_T2_pred"),
    # Win probability cards (the headline numbers)
    _card(240,  10, 240, 100, "predicted_matchups", "Selected T1 Win Prob",  "Team 1 Win Probability"),
    _card(490,  10, 240, 100, "predicted_matchups", "Selected T2 Win Prob",  "Team 2 Win Probability"),
    _card(740,  10, 240, 100, "predicted_matchups", "Avg Upset Probability", "Upset Probability"),
    _card(990,  10, 270, 100, "predicted_matchups", "Favourite Avg Win Prob","Favourite Win Prob"),
    # Full matchup table (all pairings matching current filter)
    _table_visual(
        240, 120, 1020, 240,
        "tbl_pred_matchups",
        "All Predicted Matchups (filter by Team 1 & Team 2 slicers)",
        [
            ("predicted_matchups", "T1_Seed",          False),
            ("predicted_matchups", "T1_TeamName",       False),
            ("predicted_matchups", "T1_WinProb",        False),
            ("predicted_matchups", "T2_WinProb",        False),
            ("predicted_matchups", "T2_TeamName",       False),
            ("predicted_matchups", "T2_Seed",           False),
            ("predicted_matchups", "Favourite",         False),
            ("predicted_matchups", "FavouriteWinProb",  False),
            ("predicted_matchups", "Upset_prob",        False),
            ("predicted_matchups", "MatchupLabel",      False),
        ],
    ),
    # Win probability heatmap proxy: bar chart of T1 win prob by opponent
    _clustered_bar(
        240, 370, 510, 320,
        "bar_t1_vs_all",
        "Team 1 Win Probability vs. Every Opponent",
        "predicted_matchups", "T2_TeamName",
        "predicted_matchups", ["Selected T1 Win Prob"],
    ),
    # Upset probability ranked bar
    _clustered_bar(
        760, 370, 500, 320,
        "bar_upset_by_seed",
        "Upset Probability by Seed Matchup (T1 seed vs T2 seed)",
        "predicted_matchups", "T1_seed",
        "predicted_matchups", ["Avg Upset Probability"],
    ),
]

REPORT_LAYOUT = {
    "id": 0,
    "resourcePackages": [],
    "sections": [
        {
            "id": 0,
            "name": "ReportSection1",
            "displayName": "Team Matchup",
            "width": CANVAS_W,
            "height": CANVAS_H,
            "visualContainers": page1_visuals,
            "config": json.dumps({"defaultDrillFilterOtherVisuals": True}),
        },
        {
            "id": 1,
            "name": "ReportSection2",
            "displayName": "Season Overview",
            "width": CANVAS_W,
            "height": CANVAS_H,
            "visualContainers": page2_visuals,
            "config": json.dumps({"defaultDrillFilterOtherVisuals": True}),
        },
        {
            "id": 2,
            "name": "ReportSection3",
            "displayName": "Matchup Explorer",
            "width": CANVAS_W,
            "height": CANVAS_H,
            "visualContainers": page3_visuals,
            "config": json.dumps({"defaultDrillFilterOtherVisuals": True}),
        },
    ],
    "config": json.dumps({
        "version": "5.48",
        "themeCollection": {"baseTheme": {"name": "CY23SU08", "version": "5.48"}},
        "activeSectionIndex": 0,
    }),
}

# ---------------------------------------------------------------------------
# Write .pbit (zip)
# ---------------------------------------------------------------------------
CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="json" ContentType="application/json" />
  <Default Extension="xml"  ContentType="application/xml" />
  <Override PartName="/DataModelSchema"    ContentType="application/json" />
  <Override PartName="/DiagramLayout"      ContentType="application/json" />
  <Override PartName="/Report/Layout"      ContentType="application/json" />
  <Override PartName="/Settings"           ContentType="application/json" />
  <Override PartName="/Version"            ContentType="application/json" />
</Types>"""

DIAGRAM_LAYOUT = json.dumps({
    "version": 2,
    "tables": [
        {"id": 0, "name": "matchup_facts",       "x": 0,   "y": 0,   "width": 220, "height": 300},
        {"id": 1, "name": "team_season_stats",   "x": 260, "y": 0,   "width": 220, "height": 320},
        {"id": 2, "name": "team_names",          "x": 0,   "y": 320, "width": 220, "height": 100},
        {"id": 3, "name": "seeds",               "x": 260, "y": 340, "width": 220, "height": 130},
        {"id": 4, "name": "predicted_matchups",  "x": 520, "y": 0,   "width": 220, "height": 340},
    ],
})

SETTINGS = json.dumps({
    "useStyledTooltips": True,
    "version": "5.48",
})

VERSION = json.dumps({"version": "5.48"})


def write_pbit():
    with zipfile.ZipFile(PBIT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", CONTENT_TYPES)
        zf.writestr("DataModelSchema",    json.dumps(DATA_MODEL, indent=2))
        zf.writestr("DiagramLayout",      DIAGRAM_LAYOUT)
        zf.writestr("Report/Layout",      json.dumps(REPORT_LAYOUT, indent=2))
        zf.writestr("Settings",           SETTINGS)
        zf.writestr("Version",            VERSION)
    print(f"Created: {PBIT_PATH}")


if __name__ == "__main__":
    write_pbit()
    print(
        "\nNext steps:\n"
        "  1. Run generate_powerbi_data.py to produce the four CSV files.\n"
        "  2. Open TeamMatchupsDashboard.pbit in Power BI Desktop.\n"
        "  3. When prompted for _DataFolder, enter the full path to your\n"
        f"     powerbi_data/ folder (e.g. {OUT_DIR}/).\n"
        "  4. Click Load – Power BI will import the CSVs and render the dashboard.\n"
        "\nDashboard pages:\n"
        "  • Team Matchup      – filter by Season / Team 1 / Team 2;\n"
        "                        see win probability, box-score comparison,\n"
        "                        and every historical meeting.\n"
        "  • Season Overview   – upset rates, seed win %, quality scatter,\n"
        "                        top-team win % bar, and ranking table.\n"
        "  • Matchup Explorer  – pick any two 2025 tournament teams and see\n"
        "                        the submitted model's predicted win probability\n"
        "                        for every possible pairing.\n"
    )
