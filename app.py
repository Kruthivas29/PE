import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PE as a Catalyst | GCC/MENA IBR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme / CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F9FC;
    color: #1A2742;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1B3A6B !important;
    color: white !important;
}
section[data-testid="stSidebar"] * { color: white !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label { color: #CBD5E1 !important; font-size: 0.78rem; }

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #1B3A6B 0%, #2C5282 60%, #1A3A5C 100%);
    padding: 2rem 2.5rem 1.6rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.hero-banner h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 400;
    margin: 0 0 0.3rem;
    letter-spacing: -0.01em;
    color: white;
}
.hero-banner p {
    font-size: 0.88rem;
    color: #A8C4E0;
    margin: 0;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: #CBD5E1;
    margin-bottom: 0.7rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    border-left: 4px solid #1B3A6B;
    box-shadow: 0 1px 6px rgba(27,58,107,0.07);
}
.kpi-label { font-size: 0.72rem; color: #6B7FA3; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 500; }
.kpi-value { font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: #1B3A6B; margin: 0.1rem 0; }
.kpi-sub { font-size: 0.75rem; color: #8FA3C0; }

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem;
    color: #1B3A6B;
    border-bottom: 2px solid #E2EBF6;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Hypothesis badge */
.hyp-supported { background:#E6F4EA; color:#1E6B3A; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
.hyp-rejected  { background:#FEF2F2; color:#B91C1C; border-radius:6px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

/* Tab overrides */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid #E2EBF6; }
.stTabs [data-baseweb="tab"] { font-size: 0.85rem; color: #6B7FA3; padding: 8px 18px; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"] { background: #1B3A6B !important; color: white !important; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plotly template ───────────────────────────────────────────────────────────
NAVY   = "#1B3A6B"
GOLD   = "#C9962C"
STEEL  = "#4A7DB5"
PALE   = "#A8C4E0"
COLORS = [NAVY, GOLD, STEEL, "#2ECC71", "#E74C3C", "#9B59B6", "#1ABC9C"]

def light_template():
    return dict(
        layout=go.Layout(
            font=dict(family="DM Sans", color="#1A2742"),
            paper_bgcolor="white",
            plot_bgcolor="#F7F9FC",
            gridcolor="#E8EEF7",
            title_font=dict(family="DM Serif Display", size=15, color=NAVY),
        )
    )

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    firms    = pd.read_csv("pe_firms_dataset.csv")
    deals    = pd.read_csv("gcc_pe_deal_value.csv")
    gdp      = pd.read_csv("nonoil_gdp_growth.csv")
    sectors  = pd.read_csv("sectoral_allocation.csv")
    reg      = pd.read_csv("regression_results.csv")
    desc     = pd.read_csv("descriptive_stats.csv")
    ftypes   = pd.read_csv("firm_type_summary.csv")
    return firms, deals, gdp, sectors, reg, desc, ftypes

firms, deals, gdp, sectors, reg, desc, ftypes = load()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Filters")
    st.markdown("---")

    countries = ["All"] + sorted(firms["HQ Country"].unique().tolist())
    sel_country = st.selectbox("Country", countries)

    firm_types = ["All"] + sorted(firms["Firm Type"].unique().tolist())
    sel_type = st.selectbox("Firm Type", firm_types)

    sizes = ["All"] + firms["Firm_Size_Category"].unique().tolist()
    sel_size = st.selectbox("Firm Size", sizes)

    st.markdown("---")
    st.markdown("**Research**")
    st.caption("IBR Term 2 Mid-Review")
    st.caption("MS25GF030 · Kruthivas M.T.")
    st.caption("Mentor: Farah Naaz")
    st.caption("SP Jain School of Global Mgmt")

# ── Filter firms ──────────────────────────────────────────────────────────────
df = firms.copy()
if sel_country != "All": df = df[df["HQ Country"] == sel_country]
if sel_type    != "All": df = df[df["Firm Type"]   == sel_type]
if sel_size    != "All": df = df[df["Firm_Size_Category"] == sel_size]

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">IBR Term 2 · Mid-Review Dashboard</div>
  <h1>Private Equity as a Catalyst for Economic Diversification</h1>
  <p>GCC / MENA Region &nbsp;·&nbsp; 30 PE Firms &nbsp;·&nbsp; Cross-Sectional OLS Regression &nbsp;·&nbsp; 2015–2024</p>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    ("Firms Analysed",    f"{len(df)}",              "of 30 GCC/MENA PE firms"),
    ("Total AUM",         f"${df['AUM_USD_M'].sum()/1e6:.1f}T",  "combined assets under mgmt"),
    ("Avg Non-Oil Deal Share", f"{df['NonOil_Deal_Share_Pct'].mean():.1f}%", "of portfolio in non-oil sectors"),
    ("GCC PE Deal Value", "$47.9B",                  "cumulative 2015–2024"),
    ("Strongest Finding", "H4 ✓",                    "Exit maturity p=0.022"),
]
for col, (label, val, sub) in zip([k1,k2,k3,k4,k5], kpis):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 PE Deal Trends", "🏢 Firm Analysis", "🔬 Regression Results",
    "🌍 Sectoral Shift", "📋 Data Tables"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PE Deal Trends
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">GCC Private Equity Deal Value (2015–2024)</div>', unsafe_allow_html=True)

    years = [str(y) for y in range(2015, 2025)]
    country_rows = deals[deals["Country"] != "GCC Total"]

    # Line chart by country
    fig_line = go.Figure()
    for i, row in country_rows.iterrows():
        vals = [row[y] for y in years]
        fig_line.add_trace(go.Scatter(
            x=years, y=vals, mode="lines+markers",
            name=row["Country"], line=dict(width=2.5),
            marker=dict(size=6),
        ))
    fig_line.update_layout(
        template=light_template(), height=380,
        title="Annual PE Deal Value by GCC Country (USD Millions)",
        legend=dict(orientation="h", y=-0.2),
        xaxis_title="Year", yaxis_title="Deal Value (USD M)",
        colorway=COLORS,
    )
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)

    # Bar — Total deal by country
    with c1:
        fig_bar = px.bar(
            country_rows.sort_values("Total_USD_M", ascending=True),
            x="Total_USD_M", y="Country", orientation="h",
            color="CAGR_Pct", color_continuous_scale=["#A8C4E0", NAVY],
            title="Total PE Deal Value 2015–2024 (USD M)",
            labels={"Total_USD_M": "Total (USD M)", "CAGR_Pct": "CAGR %"},
        )
        fig_bar.update_layout(template=light_template(), height=320, coloraxis_showscale=True)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Line — non-oil GDP growth
    with c2:
        gdp_melt = gdp.melt(id_vars=["Country"], value_vars=years, var_name="Year", value_name="NonOil_GDP_Growth_Pct")
        fig_gdp = px.line(
            gdp_melt, x="Year", y="NonOil_GDP_Growth_Pct", color="Country",
            title="Non-Oil GDP Growth Rate by Country (%)",
            labels={"NonOil_GDP_Growth_Pct": "Growth Rate (%)"},
            color_discrete_sequence=COLORS,
        )
        fig_gdp.update_layout(template=light_template(), height=320,
                              legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(fig_gdp, use_container_width=True)

    # Pre vs Post reform comparison
    st.markdown('<div class="section-header">Pre vs Post-Reform Non-Oil GDP Growth (2015–19 vs 2020–24)</div>', unsafe_allow_html=True)
    reform_df = gdp[["Country", "PreReform_Avg_2015_19", "PostReform_Avg_2020_24"]].copy()
    reform_melt = reform_df.melt(id_vars="Country", var_name="Period", value_name="Avg_Growth")
    reform_melt["Period"] = reform_melt["Period"].map({
        "PreReform_Avg_2015_19": "Pre-Reform (2015–19)",
        "PostReform_Avg_2020_24": "Post-Reform (2020–24)"
    })
    fig_reform = px.bar(
        reform_melt, x="Country", y="Avg_Growth", color="Period", barmode="group",
        color_discrete_map={"Pre-Reform (2015–19)": PALE, "Post-Reform (2020–24)": NAVY},
        labels={"Avg_Growth": "Avg Growth Rate (%)"},
        title="Average Non-Oil GDP Growth: Pre vs Post Vision 2030 / UAE Reform Era",
    )
    fig_reform.update_layout(template=light_template(), height=340,
                              legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_reform, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Firm Analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Firm-Level Analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Bubble chart: AUM vs Non-Oil Deal Share, sized by portfolio cos
        fig_bubble = px.scatter(
            df, x="AUM_USD_M", y="NonOil_Deal_Share_Pct",
            size="Active_NonOil_Portfolio_Cos", color="HQ Country",
            hover_name="Firm Name", log_x=True,
            title="AUM vs Non-Oil Deal Share (bubble = portfolio breadth)",
            labels={"AUM_USD_M": "AUM USD M (log)", "NonOil_Deal_Share_Pct": "Non-Oil Deal Share (%)"},
            color_discrete_sequence=COLORS,
        )
        fig_bubble.update_layout(template=light_template(), height=380)
        st.plotly_chart(fig_bubble, use_container_width=True)

    with c2:
        # Firm type comparison
        fig_type = px.bar(
            ftypes.sort_values("Mean_NonOil_Deal_Share_Pct", ascending=True),
            x="Mean_NonOil_Deal_Share_Pct", y="Firm_Type", orientation="h",
            color="Mean_NonOil_GDP_Contrib_USD_M",
            color_continuous_scale=["#E2EBF6", NAVY],
            title="Avg Non-Oil Deal Share by Firm Type",
            labels={"Mean_NonOil_Deal_Share_Pct": "Non-Oil Deal Share (%)",
                    "Mean_NonOil_GDP_Contrib_USD_M": "GDP Contrib (USD M)"},
        )
        fig_type.update_layout(template=light_template(), height=380)
        st.plotly_chart(fig_type, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        # Country-level avg non-oil contribution
        country_agg = df.groupby("HQ Country").agg(
            Firms=("Firm Name","count"),
            Avg_NonOil_GDP=("NonOil_GDP_Contrib_USD_M","mean"),
            Avg_Deal_Share=("NonOil_Deal_Share_Pct","mean"),
        ).reset_index()
        fig_ctry = px.scatter(
            country_agg, x="Avg_NonOil_GDP", y="Avg_Deal_Share",
            size="Firms", color="HQ Country", text="HQ Country",
            title="Country Avg: Non-Oil GDP Contribution vs Deal Share",
            color_discrete_sequence=COLORS,
        )
        fig_ctry.update_traces(textposition="top center")
        fig_ctry.update_layout(template=light_template(), height=340, showlegend=False)
        st.plotly_chart(fig_ctry, use_container_width=True)

    with c4:
        # Policy reform score vs sectoral focus
        fig_pf = px.scatter(
            df, x="Policy_Reform_Score", y="Sectoral_Focus_Score",
            color="HQ Country", hover_name="Firm Name",
            size="NonOil_Deal_Share_Pct",
            title="Policy Reform Score vs Sectoral Focus Score",
            labels={"Policy_Reform_Score":"Policy Reform Score (IV3)",
                    "Sectoral_Focus_Score":"Sectoral Focus Score (IV5)"},
            color_discrete_sequence=COLORS,
        )
        fig_pf.update_layout(template=light_template(), height=340)
        st.plotly_chart(fig_pf, use_container_width=True)

    # DIFC vs Non-DIFC
    st.markdown('<div class="section-header">DIFC/ADGM Domicile Impact (H2)</div>', unsafe_allow_html=True)
    difc_agg = df.groupby("DIFC_ADGM").agg(
        Avg_NonOil_GDP=("NonOil_GDP_Contrib_USD_M","mean"),
        Avg_Deal_Share=("NonOil_Deal_Share_Pct","mean"),
        Count=("Firm Name","count"),
    ).reset_index()
    difc_agg["Domicile"] = difc_agg["DIFC_ADGM"].map({1:"DIFC/ADGM", 0:"Non-DIFC"})
    dc1, dc2 = st.columns(2)
    for col, metric, label in [(dc1,"Avg_NonOil_GDP","Avg Non-Oil GDP Contribution (USD M)"),
                                (dc2,"Avg_Deal_Share","Avg Non-Oil Deal Share (%)")]:
        fig_d = px.bar(difc_agg, x="Domicile", y=metric,
                       color="Domicile", color_discrete_map={"DIFC/ADGM":NAVY,"Non-DIFC":PALE},
                       title=label, text=metric)
        fig_d.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig_d.update_layout(template=light_template(), height=300, showlegend=False)
        col.plotly_chart(fig_d, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Regression Results
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">OLS Regression Results — 4 Models</div>', unsafe_allow_html=True)

    # Hypothesis summary
    hyps = [
        ("H1","AUM (IV1)","ln(AUM) coef = +0.501, p = 0.000","Supported ✓","supported"),
        ("H2","DIFC/ADGM (IV2)","Coef = +0.063, p = 0.402","Not Supported ✗","rejected"),
        ("H3","Policy Reform Score (IV3)","Coef = +0.051, p = 0.467","Not Supported ✗","rejected"),
        ("H4","Non-Oil Portfolio Cos (IV4)","Coef = +0.028, p = 0.000 (M3)","Supported ✓","supported"),
        ("H5","Sectoral Focus Score (IV5)","Coef = +11.31, p = 0.000 for DV2","Supported ✓","supported"),
    ]
    cols = st.columns(5)
    for col, (h, var, detail, result, status) in zip(cols, hyps):
        badge = f'<span class="hyp-supported">{result}</span>' if status=="supported" else f'<span class="hyp-rejected">{result}</span>'
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:1rem;box-shadow:0 1px 6px rgba(27,58,107,0.08);height:160px;">
          <div style="font-size:1.1rem;font-weight:700;color:{NAVY};margin-bottom:4px;">{h}</div>
          <div style="font-size:0.75rem;color:#4A7DB5;margin-bottom:6px;font-weight:500;">{var}</div>
          <div style="font-size:0.7rem;color:#6B7FA3;margin-bottom:8px;">{detail}</div>
          {badge}
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # Coefficient chart — Model 1 (DV1)
        vars_plot = ["IV1_ln_AUM","IV2_DIFC_ADGM","IV3_Policy_Reform_Score",
                     "IV4_NonOil_Portfolio_Cos","IV5_Sectoral_Focus_Score"]
        labels    = ["ln(AUM)\n(H1)","DIFC/ADGM\n(H2)","Policy Score\n(H3)",
                     "Portfolio Cos\n(H4)","Sectoral Focus\n(H5)"]
        m1 = reg[reg["Variable"].isin(vars_plot)].set_index("Variable")
        coefs = [float(m1.loc[v,"Model1_DV1_Coef"]) for v in vars_plot]
        pvals = [float(m1.loc[v,"Model1_DV1_pvalue"]) for v in vars_plot]
        colors_bar = [NAVY if p < 0.05 else PALE for p in pvals]
        fig_coef = go.Figure(go.Bar(
            x=labels, y=coefs,
            marker_color=colors_bar,
            text=[f"p={p:.3f}" for p in pvals],
            textposition="outside",
        ))
        fig_coef.update_layout(
            template=light_template(), height=360,
            title="Model 1: Coefficients — DV1 Non-Oil GDP Contribution (log)",
            yaxis_title="Coefficient", xaxis_title="",
        )
        fig_coef.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        st.plotly_chart(fig_coef, use_container_width=True)

    with c2:
        # Coefficient chart — Model 2 (DV2)
        coefs2 = [float(m1.loc[v,"Model2_DV2_Coef"]) for v in vars_plot]
        pvals2 = [float(m1.loc[v,"Model2_DV2_pvalue"]) for v in vars_plot]
        colors2 = [GOLD if p < 0.05 else PALE for p in pvals2]
        fig_coef2 = go.Figure(go.Bar(
            x=labels, y=coefs2,
            marker_color=colors2,
            text=[f"p={p:.3f}" for p in pvals2],
            textposition="outside",
        ))
        fig_coef2.update_layout(
            template=light_template(), height=360,
            title="Model 2: Coefficients — DV2 Non-Oil Deal Share (%)",
            yaxis_title="Coefficient",
        )
        fig_coef2.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        st.plotly_chart(fig_coef2, use_container_width=True)

    # Model fit comparison
    st.markdown('<div class="section-header">Model Fit Statistics</div>', unsafe_allow_html=True)
    fit_data = {
        "Metric":       ["R² (Overall)", "R² (Adjusted)", "F-Statistic", "N"],
        "Model 1 (DV1 Full)":  ["0.9961", "0.9941", "313.72", "30"],
        "Model 2 (DV2 Full)":  ["0.9925", "0.9885", "155.76", "30"],
        "Model 3 (DV1 Parsimonious)": ["0.9943", "0.9931", "778.52", "30"],
        "Model 4 (DV2 Parsimonious)": ["0.9873", "0.9846", "462.38", "30"],
    }
    st.dataframe(pd.DataFrame(fit_data).set_index("Metric"), use_container_width=True)

    # Scatter: AUM vs NonOil GDP (H1 visualisation)
    st.markdown('<div class="section-header">H1 Visualised: ln(AUM) vs Non-Oil GDP Contribution</div>', unsafe_allow_html=True)
    firms_copy = firms.copy()
    firms_copy["ln_AUM"] = np.log(firms_copy["AUM_USD_M"])
    firms_copy["log_GDP"] = np.log(firms_copy["NonOil_GDP_Contrib_USD_M"])
    fig_h1 = px.scatter(
        firms_copy, x="ln_AUM", y="log_GDP",
        color="HQ Country", hover_name="Firm Name",
        trendline="ols",
        title="H1: ln(AUM) vs log(Non-Oil GDP Contribution) — OLS Trendline",
        labels={"ln_AUM":"ln(AUM)","log_GDP":"log(Non-Oil GDP Contrib)"},
        color_discrete_sequence=COLORS,
    )
    fig_h1.update_layout(template=light_template(), height=380)
    st.plotly_chart(fig_h1, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Sectoral Shift
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">PE Sectoral Allocation Shift (2015–2024)</div>', unsafe_allow_html=True)

    sect_years = ["2015","2017","2019","2021","2023","2024_Est"]
    sect_melt = sectors.melt(id_vars="Sector", value_vars=sect_years,
                             var_name="Year", value_name="Allocation_Pct")
    sect_melt["Year"] = sect_melt["Year"].str.replace("_Est","")

    # Stacked area
    fig_area = px.area(
        sect_melt, x="Year", y="Allocation_Pct", color="Sector",
        title="PE Deal Allocation by Sector — GCC/MENA 2015–2024 (%)",
        labels={"Allocation_Pct":"Allocation (%)"},
        color_discrete_sequence=COLORS,
    )
    fig_area.update_layout(template=light_template(), height=400,
                           legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_area, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        # 2015 pie
        fig_pie15 = px.pie(
            sectors, values="2015", names="Sector",
            title="Sector Mix — 2015",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig_pie15.update_layout(template=light_template(), height=340)
        st.plotly_chart(fig_pie15, use_container_width=True)

    with c2:
        # 2024 pie
        fig_pie24 = px.pie(
            sectors, values="2024_Est", names="Sector",
            title="Sector Mix — 2024 (Est.)",
            color_discrete_sequence=COLORS,
            hole=0.4,
        )
        fig_pie24.update_layout(template=light_template(), height=340)
        st.plotly_chart(fig_pie24, use_container_width=True)

    # Highlight tech growth
    tech = sectors[sectors["Sector"] == "Technology / TMT / Fintech"].iloc[0]
    oil  = sectors[sectors["Sector"] == "Oil & Gas / Traditional"].iloc[0]
    m1c, m2c, m3c = st.columns(3)
    m1c.metric("Tech/Fintech Allocation 2015", f"{tech['2015']}%")
    m1c.metric("Tech/Fintech Allocation 2024", f"{tech['2024_Est']}%", f"+{tech['2024_Est']-tech['2015']}pp")
    m2c.metric("Oil & Gas Allocation 2015", f"{oil['2015']}%")
    m2c.metric("Oil & Gas Allocation 2024", f"{oil['2024_Est']}%", f"{oil['2024_Est']-oil['2015']}pp")
    m3c.metric("Non-Oil Sectors Combined 2024", "92%", "+22pp since 2015")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Data Tables
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Full Firm Dataset</div>', unsafe_allow_html=True)
    show_cols = ["Firm Name","HQ Country","Firm Type","Founded","AUM_USD_M",
                 "NonOil_GDP_Contrib_USD_M","NonOil_Deal_Share_Pct",
                 "Policy_Reform_Score","Sectoral_Focus_Score","Firm_Size_Category"]
    st.dataframe(
        df[show_cols].rename(columns={
            "AUM_USD_M":"AUM (USD M)",
            "NonOil_GDP_Contrib_USD_M":"Non-Oil GDP Contrib (USD M)",
            "NonOil_Deal_Share_Pct":"Non-Oil Deal Share (%)",
            "Policy_Reform_Score":"Policy Score",
            "Sectoral_Focus_Score":"Sectoral Focus",
        }).style.background_gradient(subset=["Non-Oil Deal Share (%)"], cmap="Blues"),
        use_container_width=True, height=450,
    )

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(desc.set_index("Variable"), use_container_width=True)

    st.markdown('<div class="section-header">Correlation Matrix Highlights</div>', unsafe_allow_html=True)
    corr_data = {
        "Variable Pair":["ln(AUM) ↔ log(Non-Oil GDP)","Sectoral Focus ↔ Non-Oil Deal Share",
                         "Policy Score ↔ Non-Oil Deal Share","IV4 Portfolio Cos ↔ log(Non-Oil GDP)",
                         "Cumulative Exits ↔ log(Non-Oil GDP)"],
        "r":["0.973","0.991","0.709","0.818","0.585"],
        "Strength":["Strong ●","Strong ●","Strong ●","Strong ●","Moderate ◑"],
    }
    st.dataframe(pd.DataFrame(corr_data).set_index("Variable Pair"), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#8FA3C0;font-size:0.75rem;'>"
    "IBR Term 2 Mid-Review · MS25GF030 · Kruthivas Mahesh Tambralli · "
    "SP Jain School of Global Management · Mentor: Farah Naaz</p>",
    unsafe_allow_html=True
)
