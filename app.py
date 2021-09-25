import json
from typing import Dict, List, Set

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


@st.cache
def _vhver() -> str:
    GAME_VERSION = {
        "major": 0,
        "minor": 202,
        "patch": 19,
    }
    return ".".join(str(v) for v in GAME_VERSION.values())


@st.cache
def _load_food_json() -> Dict:
    """Load food data from JSON into a Python dictionary."""
    with open("data/food.json", "r") as f:
        return json.load(f)


@st.cache
def load_food_df() -> pd.DataFrame:
    """Load food data from JSON into Pandas DataFrame."""
    df = pd.read_json("data/food.json", orient="records")
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
    data = _load_food_json()
    rec = [None] * len(data)
    for i, d in enumerate(data):
        name, amount, ingr = d["name"], d["amount"], d["ingredients"]
        if len(ingr) < 2 and list(ingr.values())[0] < 2:
            continue
        rec[i] = (
            f"{name.upper()}"
            + (f" x{amount}" if amount > 1 else "")
            + "\n"
            + "\n".join(f"{k}: {v}" for k, v in ingr.items())
            + "\n"
        )
    return list(filter(None, rec))


# ----- Streamlit Layout and Formatting -----


def st_init():
    """Initialize the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Valheim Food Statistics")
    st.caption(f"Valheim Version: {_vhver()}")


def st_recipes(nc):
    """Write all food recipes."""
    cols = st.columns(nc)
    rec = recipes()
    nr = len(rec)
    npc = (nr + nc) // nc
    for i in range(nr):
        cols[i // npc].text(rec[i])


# ----- Plots -----


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
        barmode="stack",
        # Show each food item as single datum in stacked bar chart.
        hovermode="x",
        # Reduce empty space around chart area.
        margin={
            "b": 20,
            "l": 0,
            "r": 0,
            "t": 20,
        },
        # Bar charts shouldn't be too tall.
        height=400,
    )
    # Descriptive axis labels.
    fig.update_yaxes(title_text="Health & Stamina", showgrid=False, secondary_y=False)
    fig.update_yaxes(title_text="Duration (min)", showgrid=False, secondary_y=True)
    # Always show all food items on the x-axis.
    fig.update_xaxes(tickmode="linear")
    return fig
    return fig


# ----- Main -----


def main():
    """Main application function."""

    # Initialize document.
    st_init()

    # Load data.
    df = load_food_df()

    # Ingredient filtering.
    select_ing = st.sidebar.multiselect(
        "Ingredients", get_ingredients(df), get_ingredients(df)
    )
    ingredient_choice = ingredient_mask(df, select_ing)

    st.header("List of Food Stats")
    st.dataframe(
        # FIXME drop column since style.hide_columns doesn't work
        data=df[ingredient_choice]
        .drop(columns=["amount", "ingredients"])
        .style.bar(subset=["health"], color="#636efa")
        .bar(subset=["stamina"], color="#ef553b")
        .bar(subset=["total"], color="#ab63fa")
        .bar(subset=["duration"], color="#00cc96"),
        height=1000,
    )

    st.header("Food Ranking")
    st.plotly_chart(
        plotly_hsd(df[ingredient_choice]),
        config={
            "displayModeBar": False,
        },
    )

    # TODO: Style with custom CSS to display nicely in discrete boxes.
    st.header("Recipes")
    with st.container():
        st_recipes(3)


if __name__ == "__main__":
    main()
