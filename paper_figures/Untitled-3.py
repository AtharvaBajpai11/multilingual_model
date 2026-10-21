"""
Lab: Profiling a Large Corporate Dataset with pandas
=====================================================
AIM
    Use pandas to programmatically load and profile a large corporate dataset,
    extract NON-GRAPHICAL summary statistics, mathematically handle missing
    values, and identify extreme anomalies (outliers).

USAGE
    python corporate_data_profiling.py                  # uses synthetic data
    python corporate_data_profiling.py my_data.csv      # uses your own CSV

Requires: pandas, numpy   (scipy is optional, used for a chi-square cutoff)
"""

import sys
import numpy as np
import pandas as pd

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")


def banner(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
# STEP 0 - Data source (synthetic corporate HR/finance data if no CSV given)
# ----------------------------------------------------------------------------
def make_synthetic(n=200_000, seed=42):
    rng = np.random.default_rng(seed)
    dept = rng.choice(["Sales", "Engineering", "Finance", "HR", "Operations"],
                      n, p=[.30, .30, .15, .10, .15])
    level = rng.choice([1, 2, 3, 4, 5], n, p=[.35, .30, .20, .10, .05])
    region = rng.choice(["North", "South", "East", "West"], n)
    age = np.clip(rng.normal(37, 9, n), 20, 65).round()
    tenure = np.clip((age - 21) * rng.uniform(0.05, 0.6, n), 0, None).round(1)

    # salary is log-normal (right skewed) and depends on level and tenure
    salary = np.exp(rng.normal(10.6 + 0.28 * level + 0.02 * tenure, 0.20))
    bonus = salary * rng.uniform(0.02, 0.15, n)
    perf = np.clip(rng.normal(3.2, 0.7, n), 1, 5).round(2)
    hours = np.clip(rng.normal(42, 5, n), 20, 70).round(1)
    expense = rng.gamma(2.0, 400.0, n)

    df = pd.DataFrame(dict(employee_id=np.arange(1, n + 1), department=dept,
                           job_level=level, region=region, age=age,
                           tenure_years=tenure, salary=salary, bonus=bonus,
                           performance_score=perf, weekly_hours=hours,
                           expense_claims=expense))

    # --- inject missing values -------------------------------------------
    df.loc[rng.random(n) < 0.05, "salary"] = np.nan             # MCAR
    df.loc[(df.department == "Sales") & (rng.random(n) < 0.30), "bonus"] = np.nan  # MAR
    df.loc[rng.random(n) < 0.08, "performance_score"] = np.nan
    df.loc[rng.random(n) < 0.03, "age"] = np.nan
    df.loc[rng.random(n) < 0.02, "region"] = np.nan

    # --- inject extreme anomalies ----------------------------------------
    idx = rng.choice(n, 150, replace=False)
    df.loc[idx[:50], "salary"] *= 12             # data-entry error (extra zero+)
    df.loc[idx[50:100], "expense_claims"] *= 40  # fraud-like claims
    df.loc[idx[100:120], "weekly_hours"] = 160   # impossible (> 168 h/week region)
    df.loc[idx[120:140], "tenure_years"] = -5    # negative tenure
    df.loc[idx[140:], "age"] = 140               # impossible age
    df = pd.concat([df, df.sample(300, random_state=1)], ignore_index=True)  # duplicates
    return df


def load_data(path=None):
    if path:
        return pd.read_csv(path)
    return make_synthetic()


# ----------------------------------------------------------------------------
# STEP 1 - Structural profile
# ----------------------------------------------------------------------------
def structural_profile(df):
    banner("1. STRUCTURAL PROFILE")
    print(f"Rows x Columns      : {df.shape[0]:,} x {df.shape[1]}")
    print(f"Memory usage (MB)   : {df.memory_usage(deep=True).sum() / 1e6:,.2f}")
    print(f"Duplicate rows      : {df.duplicated().sum():,}")
    print(f"Duplicate IDs       : {df['employee_id'].duplicated().sum():,}"
          if "employee_id" in df else "")
    print("\nColumn dtypes and cardinality:")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str),
                         "non_null": df.notna().sum(),
                         "n_unique": df.nunique()})
    print(info)


# ----------------------------------------------------------------------------
# STEP 2 - Non-graphical summary statistics
# ----------------------------------------------------------------------------
def numeric_summary(df):
    banner("2. NON-GRAPHICAL SUMMARY STATISTICS")
    num = df.select_dtypes("number").drop(columns=["employee_id"], errors="ignore")

    desc = num.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc["range"] = desc["max"] - desc["min"]
    desc["IQR"] = desc["75%"] - desc["25%"]
    desc["variance"] = num.var()
    desc["CV"] = desc["std"] / desc["mean"]          # coefficient of variation
    desc["skewness"] = num.skew()                    # 3rd standardised moment
    desc["kurtosis"] = num.kurt()                    # excess kurtosis (normal = 0)
    desc["median"] = num.median()
    desc["mode"] = num.mode().iloc[0]
    print(desc.T)

    banner("2b. CATEGORICAL SUMMARY")
    for col in df.select_dtypes(exclude="number").columns:
        vc = df[col].value_counts(dropna=False, normalize=True) * 100
        print(f"\n{col}  (% share):\n{vc.round(2).to_string()}")

    banner("2c. CORRELATION (Pearson) - NUMERIC")
    print(num.corr().round(3))

    banner("2d. GROUP-WISE STATISTICS (salary by department)")
    print(df.groupby("department")["salary"]
            .agg(["count", "mean", "median", "std", "min", "max"]))
    return num


# ----------------------------------------------------------------------------
# STEP 3 - Missing value analysis and treatment
# ----------------------------------------------------------------------------
def missing_report(df):
    banner("3. MISSING-VALUE REPORT")
    miss = pd.DataFrame({"n_missing": df.isna().sum(),
                         "pct_missing": df.isna().mean() * 100})
    miss = miss[miss.n_missing > 0].sort_values("pct_missing", ascending=False)
    print(miss)
    return miss


def handle_missing(df):
    """
    Mathematical imputation rules
    ------------------------------
    * Mean       : x_hat = (1/n) * sum(x_i)           -> symmetric data (|skew| < 1)
    * Median     : x_hat = middle order statistic     -> skewed data (|skew| >= 1)
    * Group      : x_hat = median(x | department, job_level)   -> uses structure
    * Regression : bonus_hat = b0 + b1*salary   (b = (X'X)^-1 X'y via least squares)
    * Categorical: mode
    Imputation by a constant SHRINKS variance, so we verify Var before/after.
    """
    banner("4. MISSING-VALUE TREATMENT")
    out = df.copy()
    before_std = out.select_dtypes("number").std()

    # keep a flag so downstream models know which values were imputed
    for c in out.columns[out.isna().any()]:
        out[f"{c}_was_missing"] = out[c].isna().astype(np.int8)

    # (a) skew-based choice: mean vs median
    for col in ["age", "performance_score"]:
        sk = out[col].skew()
        fill = out[col].median() if abs(sk) >= 1 else out[col].mean()
        out[col] = out[col].fillna(fill)
        print(f"{col:<18} skew={sk:6.3f} -> {'median' if abs(sk) >= 1 else 'mean'} "
              f"imputation (value = {fill:,.3f})")

    # (b) group-wise median for salary
    grp_med = out.groupby(["department", "job_level"])["salary"].transform("median")
    n_before = out["salary"].isna().sum()
    out["salary"] = out["salary"].fillna(grp_med)
    print(f"{'salary':<18} group-median (department x job_level): "
          f"{n_before:,} values filled")

    # (c) regression imputation for bonus using least squares.
    #     Fit ONLY on rows whose salary is not an extreme outlier (otherwise a
    #     single 12x data-entry error drags the line), and cap salary at the
    #     Tukey fence when predicting.
    q1, q3 = out["salary"].quantile([.25, .75])
    upper = q3 + 3.0 * (q3 - q1)
    sal_ok = out["salary"].clip(upper=upper)
    known = out["bonus"].notna() & (out["salary"] <= upper)
    X = np.column_stack([np.ones(known.sum()), sal_ok[known]])
    y = out.loc[known, "bonus"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    miss = out["bonus"].isna()
    out.loc[miss, "bonus"] = beta[0] + beta[1] * sal_ok[miss]
    print(f"{'bonus':<18} regression  bonus = {beta[0]:,.1f} + {beta[1]:.4f} * salary "
          f"({miss.sum():,} values filled)")

    # (d) categorical -> mode
    mode_val = out["region"].mode()[0]
    out["region"] = out["region"].fillna(mode_val)
    print(f"{'region':<18} mode imputation -> '{mode_val}'")

    after_std = out.select_dtypes("number").std()
    cmp = pd.DataFrame({"std_before": before_std, "std_after": after_std})
    cmp["change_%"] = (cmp.std_after / cmp.std_before - 1) * 100
    print("\nVariance-shrinkage check (std before vs after):")
    print(cmp.dropna().loc[["age", "salary", "bonus", "performance_score"]])
    print(f"\nRemaining missing values: {int(out.isna().sum().sum())}")
    return out


# ----------------------------------------------------------------------------
# STEP 4 - Extreme anomaly detection
# ----------------------------------------------------------------------------
def zscore_flags(s, thr=3.0):
    """z_i = (x_i - mean) / std     flag if |z_i| > thr"""
    z = (s - s.mean()) / s.std(ddof=1)
    return z.abs() > thr


def modified_z_flags(s, thr=3.5):
    """Robust: M_i = 0.6745 * (x_i - median) / MAD, MAD = median(|x - median|)
    (Iglewicz & Hoaglin).  Flag if |M_i| > 3.5"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0:
        return pd.Series(False, index=s.index)
    return (0.6745 * (s - med) / mad).abs() > thr


def iqr_flags(s, k=3.0):
    """Tukey fences: [Q1 - k*IQR , Q3 + k*IQR]
    k = 1.5 -> mild outlier,  k = 3.0 -> EXTREME outlier"""
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def mahalanobis_flags(frame, alpha=0.001):
    """D^2 = (x - mu)' S^-1 (x - mu) ~ chi-square(p)  under multivariate normality.
    Flag if D^2 exceeds the (1 - alpha) quantile."""
    X = frame.to_numpy(dtype=float)
    diff = X - X.mean(axis=0)
    inv = np.linalg.pinv(np.cov(X, rowvar=False))
    d2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    try:
        from scipy.stats import chi2
        cutoff = chi2.ppf(1 - alpha, df=X.shape[1])
    except ImportError:
        cutoff = np.quantile(d2, 1 - alpha)
    return pd.Series(d2 > cutoff, index=frame.index), cutoff


def detect_anomalies(df, cols):
    banner("5. EXTREME ANOMALY DETECTION (univariate; log1p applied if |skew| > 1)")
    rows, votes = [], pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c]
        if abs(s.skew()) > 1:          # heavy right tail -> work on log scale
            s = np.log1p(s.clip(lower=0))
        f_z, f_m, f_i = zscore_flags(s), modified_z_flags(s), iqr_flags(s, 3.0)
        rows.append({"column": c,
                     "z>3": f_z.sum(), "modZ>3.5": f_m.sum(),
                     "IQR*3": f_i.sum(),
                     "any(>=2 agree)": ((f_z.astype(int) + f_m + f_i) >= 2).sum()})
        votes[c] = (f_z.astype(int) + f_m + f_i) >= 2
    print(pd.DataFrame(rows).set_index("column"))

    banner("5b. DOMAIN-RULE VIOLATIONS (impossible values)")
    rules = {
        "age outside [18, 80]": ~df["age"].between(18, 80),
        "tenure_years < 0": df["tenure_years"] < 0,
        "weekly_hours > 100": df["weekly_hours"] > 100,
        "salary <= 0": df["salary"] <= 0,
    }
    for name, mask in rules.items():
        print(f"{name:<26}: {int(mask.sum()):,}")

    banner("5c. MULTIVARIATE ANOMALIES (Mahalanobis distance)")
    flags, cutoff = mahalanobis_flags(df[cols])
    print(f"chi-square cutoff (alpha=0.001, p={len(cols)}): {cutoff:,.2f}")
    print(f"Flagged records: {int(flags.sum()):,}")

    consensus = votes.any(axis=1) | flags
    print(f"\nTOTAL records flagged as extreme anomalies: {int(consensus.sum()):,} "
          f"({consensus.mean() * 100:.3f}%)")
    return consensus, votes


def anomaly_impact(df, cols, flagged):
    banner("6. IMPACT OF ANOMALIES ON SUMMARY STATISTICS")
    raw = df[cols].agg(["mean", "std", "max"]).T
    clean = df.loc[~flagged, cols].agg(["mean", "std", "max"]).T
    cmp = raw.join(clean, lsuffix="_with", rsuffix="_without")
    print(cmp)

    banner("6b. WINSORISING (cap at 1st / 99th percentile) - alternative to deletion")
    wins = df[cols].clip(df[cols].quantile(.01), df[cols].quantile(.99), axis=1)
    print(wins.agg(["mean", "std", "max"]).T)


# ----------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    df = load_data(path)

    structural_profile(df)
    numeric_summary(df)
    missing_report(df)

    df = df.drop_duplicates()
    clean = handle_missing(df)

    cols = ["salary", "bonus", "expense_claims", "weekly_hours", "age"]
    flagged, _ = detect_anomalies(clean, cols)
    anomaly_impact(clean, cols, flagged)

    clean.to_csv("cleaned_dataset.csv", index=False)
    clean[flagged].to_csv("extreme_anomalies.csv", index=False)
    banner("DONE - wrote cleaned_dataset.csv and extreme_anomalies.csv")


if __name__ == "__main__":
    main()