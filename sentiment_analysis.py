
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import Patch
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.facecolor": "#0f0f0f",
                     "axes.facecolor": "#1a1a2e", "axes.labelcolor": "white",
                     "xtick.color": "white", "ytick.color": "white",
                     "text.color": "white", "grid.color": "#2e2e4e"})
SENTIMENT_COLORS = {
    "Extreme Fear":  "#e74c3c",
    "Fear":          "#e67e22",
    "Neutral":       "#f1c40f",
    "Greed":         "#2ecc71",
    "Extreme Greed": "#27ae60",
}
CATEGORY_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
def load_data(filepath: str) -> pd.DataFrame:
    """Load and return raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"\n{'='*55}")
    print(f"  Loaded: {filepath}")
    print(f"  Shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"{'='*55}")
    return df
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df["classification"] = (
        df["classification"]
        .str.strip()
        .str.title()
    )
    before = len(df)
    df.drop_duplicates(subset=["date"], keep="last", inplace=True)
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate date rows.")

    df.dropna(inplace=True)

    df["classification"] = pd.Categorical(
        df["classification"], categories=CATEGORY_ORDER, ordered=True
    )
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["month_name"]  = df["date"].dt.strftime("%b")
    df["day_of_week"] = df["date"].dt.day_name()
    df["rolling_30d"] = df["value"].rolling(30, min_periods=1).mean()
    df["value_lag1"]  = df["value"].shift(1)
    df["value_change"]= df["value"] - df["value_lag1"]
    print("\n  Preprocessed dataset info:")
    print(df.dtypes)
    return df
def print_eda(df: pd.DataFrame) -> None:
    """Print key statistical insights to console."""
    print(f"\n{'─'*55}")
    print("  CORE EDA INSIGHTS")
    print(f"{'─'*55}")

    print(f"\n  Date range   : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"  Total records: {len(df):,}")
    print(f"\n  Fear & Greed Value Statistics:")
    print(df["value"].describe().to_string())

    print(f"\n  Sentiment Distribution:")
    dist = (df["classification"]
            .value_counts()
            .reindex(CATEGORY_ORDER)
            .rename("count"))
    dist["pct"] = (dist["count"] / dist["count"].sum() * 100).round(2)
    print(dist.to_string())
    print(f"\n  Average Value per Sentiment:")
    print(df.groupby("classification", observed=True)["value"]
            .mean().round(2).to_string())
    print(f"\n  Year-wise average sentiment value:")
    print(df.groupby("year")["value"].mean().round(2).to_string())

    print(f"\n  Most Fearful date  : {df.loc[df['value'].idxmin(), 'date'].date()}  "
          f"(value={df['value'].min()})")
    print(f"  Most Greedy date   : {df.loc[df['value'].idxmax(), 'date'].date()}  "
          f"(value={df['value'].max()})")

    df["streak_id"] = (df["classification"] != df["classification"].shift()).cumsum()
    streak_len = df.groupby(["streak_id", "classification"], observed=True).size().reset_index(name="length")
    idx_max = streak_len["length"].idxmax()
    print(f"\n  Longest consecutive streak:")
    print(f"    {streak_len.loc[idx_max, 'classification']}  "
          f"for {streak_len.loc[idx_max, 'length']} days")

    print(f"\n  Day-of-week with highest avg greed:")
    dow = df.groupby("day_of_week")["value"].mean().sort_values(ascending=False)
    print(f"    {dow.index[0]}  (avg={dow.iloc[0]:.2f})")

    print(f"\n{'─'*55}\n")


def plot_time_series(df: pd.DataFrame) -> None:
    """Line chart of the Fear & Greed index over time."""
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle("Bitcoin Fear & Greed Index — Full History", fontsize=14, color="white")

    for cat in CATEGORY_ORDER:
        sub = df[df["classification"] == cat]
        ax.scatter(sub["date"], sub["value"],
                   color=SENTIMENT_COLORS[cat], s=2, alpha=0.6, label=cat)

    ax.plot(df["date"], df["rolling_30d"],
            color="white", linewidth=1.5, label="30-day MA")

    ax.axhspan(0,  25, alpha=0.08, color="#e74c3c")
    ax.axhspan(25, 45, alpha=0.08, color="#e67e22")
    ax.axhspan(45, 55, alpha=0.08, color="#f1c40f")
    ax.axhspan(55, 75, alpha=0.08, color="#2ecc71")
    ax.axhspan(75, 100,alpha=0.08, color="#27ae60")

    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Date");  ax.set_ylabel("Fear & Greed Value")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=SENTIMENT_COLORS[c], label=c) for c in CATEGORY_ORDER]
    handles.append(plt.Line2D([0],[0], color="white", linewidth=1.5, label="30-day MA"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame) -> None:
    """Sentiment category counts + value histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentiment Distribution", fontsize=13, color="white")

    counts = df["classification"].value_counts().reindex(CATEGORY_ORDER)
    colors = [SENTIMENT_COLORS[c] for c in CATEGORY_ORDER]
    axes[0].bar(CATEGORY_ORDER, counts.values, color=colors, edgecolor="black")
    axes[0].set_title("Frequency of Each Sentiment Category")
    axes[0].set_xlabel("Sentiment");  axes[0].set_ylabel("Number of Days")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 10, str(v), ha="center", fontsize=9, color="white")

    axes[1].hist(df["value"], bins=50, color="#4a90d9", edgecolor="black", alpha=0.85)
    axes[1].axvline(df["value"].mean(), color="yellow",
                    linestyle="--", label=f"Mean={df['value'].mean():.1f}")
    axes[1].axvline(df["value"].median(), color="orange",
                    linestyle=":", label=f"Median={df['value'].median():.1f}")
    axes[1].set_title("Distribution of Fear & Greed Values")
    axes[1].set_xlabel("Value (0=Extreme Fear, 100=Extreme Greed)")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_yearly_heatmap(df: pd.DataFrame) -> None:
    """Monthly average sentiment heatmap by year."""
    pivot = (df.groupby(["year", "month"]) ["value"]
               .mean()
               .unstack(level="month"))
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                     "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn",
                vmin=0, vmax=100, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Avg Fear & Greed Value"})
    ax.set_title("Monthly Average Fear & Greed Value by Year", fontsize=13)
    ax.set_xlabel("Month");  ax.set_ylabel("Year")
    plt.tight_layout()
    plt.show()


def plot_yearly_trend(df: pd.DataFrame) -> None:
    """Year-wise box plots showing sentiment spread."""
    fig, ax = plt.subplots(figsize=(14, 5))
    years = sorted(df["year"].unique())
    data_by_year = [df.loc[df["year"] == y, "value"].values for y in years]

    bp = ax.boxplot(data_by_year, labels=years, patch_artist=True,
                    medianprops=dict(color="yellow", linewidth=2))
    palette = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(years)))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)

    ax.set_title("Year-wise Fear & Greed Value Distribution", fontsize=13)
    ax.set_xlabel("Year");  ax.set_ylabel("Fear & Greed Value")
    ax.axhline(50, color="white", linestyle="--", linewidth=0.8, label="Neutral (50)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_day_of_week(df: pd.DataFrame) -> None:
    """Average sentiment by day of week."""
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = df.groupby("day_of_week")["value"].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.RdYlGn(dow.values / 100)
    ax.bar(dow.index, dow.values, color=colors, edgecolor="black")
    ax.set_title("Average Fear & Greed Value by Day of Week", fontsize=13)
    ax.set_xlabel("Day of Week");  ax.set_ylabel("Avg Value")
    ax.axhline(50, color="white", linestyle="--", linewidth=0.8)
    for i, v in enumerate(dow.values):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9, color="white")
    plt.tight_layout()
    plt.show()


def plot_sentiment_transitions(df: pd.DataFrame) -> None:
    """
    Transition matrix — given today's sentiment, what is tomorrow's?
    Shows probability as a heat map.
    """
    df2 = df.copy()
    df2["next_class"] = df2["classification"].shift(-1)
    df2.dropna(subset=["next_class"], inplace=True)

    matrix = pd.crosstab(
        df2["classification"], df2["next_class"],
        normalize="index"
    ).reindex(index=CATEGORY_ORDER, columns=CATEGORY_ORDER, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.5, ax=ax,
                cbar_kws={"label": "Transition Probability"})
    ax.set_title("Sentiment Transition Matrix (Day → Next Day)", fontsize=12)
    ax.set_xlabel("Next Day Sentiment");  ax.set_ylabel("Current Day Sentiment")
    plt.tight_layout()
    plt.show()


def plot_rolling_volatility(df: pd.DataFrame) -> None:
    """30-day rolling standard deviation of the sentiment index."""
    df2 = df.copy()
    df2["rolling_std"] = df2["value"].rolling(30, min_periods=10).std()

    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Fear & Greed Index & Rolling Volatility (30-day)", fontsize=13)

    axes[0].plot(df2["date"], df2["value"], color="#4a90d9", linewidth=0.8)
    axes[0].plot(df2["date"], df2["rolling_30d"], color="yellow",
                 linewidth=1.5, label="30-day MA")
    axes[0].set_ylabel("F&G Value");  axes[0].legend()

    axes[1].fill_between(df2["date"], df2["rolling_std"],
                         alpha=0.7, color="#e74c3c")
    axes[1].set_ylabel("30-day Std Dev")
    axes[1].set_xlabel("Date")

    plt.tight_layout()
    plt.show()


def main():
    filepath = "fear_greed_index.csv"
    df_raw = load_data(filepath)

    df = preprocess(df_raw)

    print_eda(df)

    print("  Generating charts… close each window to proceed.\n")

    plot_time_series(df)
    plot_distribution(df)
    plot_yearly_heatmap(df)
    plot_yearly_trend(df)
    plot_day_of_week(df)
    plot_sentiment_transitions(df)
    plot_rolling_volatility(df)

    print("\nsentiment charts displayed successfully.")

