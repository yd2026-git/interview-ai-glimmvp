# analytics.py
#
# Computes historical conversion metrics for the last 5 years
# Reads all CSVs placed inside: data/placement_history/
#
# Output:
#   Prints:
#       - Per-year shortlist rate
#       - Per-year PPI rate
#       - 5-year aggregate stats
#       - Company-wise conversion rates
#
#   Saves results to:
#       data/placement_history/analytics_summary.json
#

import pandas as pd
import glob
import json
import os

DATA_PATH = "data/placement_history"

def load_data():
    files = glob.glob(f"{DATA_PATH}/*.csv")
    if not files:
        raise FileNotFoundError("No CSV files found in data/placement_history/")
    df_list = []
    for f in files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
    return pd.concat(df_list, ignore_index=True)

def compute_metrics(df):
    summary = df.groupby("batch_year").agg(
        total_applications=("student_id", "count"),
        shortlist_count=("shortlisted", "sum"),
        ppi_count=("ppi", "sum")
    ).reset_index()

    summary["shortlist_rate"] = round(summary["shortlist_count"] /
                                      summary["total_applications"] * 100, 2)
    summary["ppi_rate"] = round(summary["ppi_count"] /
                                summary["total_applications"] * 100, 2)

    # 5-year aggregate
    total_apps_5yr = summary["total_applications"].sum()
    total_ppi_5yr = summary["ppi_count"].sum()
    total_shortlist_5yr = summary["shortlist_count"].sum()

    aggregate_stats = {
        "five_year_total_applications": int(total_apps_5yr),
        "five_year_total_ppi": int(total_ppi_5yr),
        "five_year_total_shortlists": int(total_shortlist_5yr),
        "five_year_ppi_rate": round((total_ppi_5yr / total_apps_5yr) * 100, 2),
        "five_year_shortlist_rate": round((total_shortlist_5yr / total_apps_5yr) * 100, 2),
    }

    # Company-wise conversion
    company_summary = df.groupby("applied_company").agg(
        applications=("student_id", "count"),
        shortlists=("shortlisted", "sum"),
        ppi=("ppi", "sum")
    ).reset_index()

    company_summary["shortlist_rate"] = round(company_summary["shortlists"] /
                                              company_summary["applications"] * 100, 2)
    company_summary["ppi_rate"] = round(company_summary["ppi"] /
                                        company_summary["applications"] * 100, 2)

    return summary, aggregate_stats, company_summary


def save_results(yearly, aggregate, company):
    out = {
        "yearly_summary": yearly.to_dict(orient="records"),
        "aggregate_5_year": aggregate,
        "company_wise": company.to_dict(orient="records")
    }

    out_path = os.path.join(DATA_PATH, "analytics_summary.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    print(f"\nAnalytics saved → {out_path}")


if __name__ == "__main__":
    df = load_data()
    yearly, agg, company = compute_metrics(df)
    print("\n===== YEARLY SUMMARY =====")
    print(yearly)

    print("\n===== AGGREGATE 5-YEAR METRICS =====")
    print(agg)

    print("\n===== COMPANY-WISE METRICS =====")
    print(company)

    save_results(yearly, agg, company)
