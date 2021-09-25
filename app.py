import json
from typing import Dict, List, Set

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def load_food_json() -> Dict:
    """Load food data from JSON into a Python dictionary."""
    with open("data/food.json", "r") as f:
        return json.load(f)


def load_food_df() -> pd.DataFrame:
    """Load food data from JSON into Pandas DataFrame."""
    return pd.read_json("data/food.json", orient="records")


@st.cache
def add_total(df: pd.DataFrame) -> pd.DataFrame:
    """Add total food benefit; health and stamina."""
    df.insert(5, "total", df["health"] + df["stamina"])
    return df.sort_values(by="total").reset_index(drop=True)


@st.cache
def get_ingredients(df: pd.DataFrame) -> Set[str]:
    """Get set of all food ingredients."""
    return set(sorted([k for d in df.ingredients for k in d.keys()]))


def ingredient_mask(df: pd.DataFrame, selection: List[str]) -> pd.DataFrame:
    """Get logical DataFrame mask rows containing selected ingredients."""
    return df.apply(
        lambda x: all(k in selection for k in x["ingredients"].keys()),
        axis=1,
    )


@st.cache
def recipes() -> List[str]:
    """Get list of formatted recipes for each food item."""
    data = load_food_json()
    rec = [None] * len(data)
    for i, d in enumerate(data):
        if len(d["ingredients"]) < 2:
            continue
        rec[i] = (
            f"{d['name'].upper()}"
            + (f" x{d['amount']}" if d["amount"] > 1 else "")
            + "\n"
            + "\n".join(f"{k}: {v}" for k, v in d["ingredients"].items())
            + "\n"
        )
    return list(filter(None, rec))


def st_init():
    """Initialize the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Valheim Food Statistics")
    st.caption("Valheim Version: 0.202.19")


def st_recipes(nc):
    """Write all food recipes."""
    cols = st.columns(nc)
    rec = recipes()
    nr = len(rec)
    npc = (nr + nc) // nc
    for i in range(nr):
        cols[i // npc].text(rec[i])


def plotly_hsd(df):
    """Plot food health, stamina, and duration in stacked bar chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=df["name"], y=df["health"], name="Health"), secondary_y=False
    )
    fig.add_trace(
        go.Bar(x=df["name"], y=df["stamina"], name="Stamina"), secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df["name"],
            y=df["duration"],
            name="Duration",
            mode="markers",
            hoverinfo="skip",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        barmode="stack", title_text="Food Ranking", width=1000, height=500
    )
    # Descriptive axis labels.
    fig.update_yaxes(title_text="Health & Stamina", secondary_y=False)
    fig.update_yaxes(title_text="Duration (min)", secondary_y=True)
    # Show all parameters per food item on hover: health, stamina, and duration.
    fig.update_layout(hovermode="x")
    return fig


def main():
    """Main application function."""

    # Initialize document.
    st_init()

    # Load data.
    df = add_total(load_food_df())

    # Ingredient filtering.
    select_ing = st.sidebar.multiselect(
        "Ingredients", get_ingredients(df), get_ingredients(df)
    )
    select_mask = ingredient_mask(df, select_ing)

    st.dataframe(
        # FIXME drop column since style.hide_columns doesn't work
        data=df[select_mask]
        .drop(columns=["amount", "ingredients"])
        .style.bar(subset=["health"], color="#636efa")
        .bar(subset=["stamina"], color="#ef553b")
        .bar(subset=["total"], color="#ab63fa")
        .bar(subset=["duration"], color="#00cc96"),
        width=1000,
        height=1000,
    )
    st.plotly_chart(plotly_hsd(df[select_mask]), width=1000)
    # TODO: Style with custom CSS to display nicely in discrete boxes.
    st.header("Recipes")
    with st.container():
        st_recipes(3)


if __name__ == "__main__":
    main()
