from src import Helper,Custom
import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import plotly.express as px
import plotly.figure_factory as ff
import time
import yfinance as yf
import concurrent.futures

st.set_page_config(layout="wide", page_title="Risk Dashboard")

PRIMARY_COLOR = "#0072B5"
SECONDARY_COLOR = "#B54300"

helper_instance = Helper.HELPER()
custom_instance = Custom.Custom_optimize()

@st.cache_data
def Bl_input_data(edited_df):
    views = dict(
        zip(
            edited_df.loc[edited_df["Use_View"], "Asset"], 
            edited_df.loc[edited_df["Use_View"], "Expected_Return"]
        )
    )
    return views


@st.cache_data
def get_data(tickers, period,max_stale_days: int = 61, max_flat_days: int = 61):
    df = helper_instance.data_downloader(tickers, period, "1d")
    df = df.dropna(axis=1, thresh=int(len(df) * 0.9))
    df = df.dropna()
    close = df["Close"].astype(int)
    last_date = close.index[-1]
    today = pd.Timestamp.now(tz=last_date.tzinfo)
    stale_cols = [
        col for col in close.columns
        if (today - close[col].last_valid_index()).days > max_stale_days
    ]
    
    flat_cols = [
        col for col in close.columns
        if (close[col].tail(max_flat_days) == close[col].tail(max_flat_days).iloc[0]).all()
    ]
    
    drop_cols = set(stale_cols) | set(flat_cols)
    if drop_cols:
        print(f"Dropping stale/flat tickers: {drop_cols}")
        close = close.drop(columns=drop_cols)
    
    return close



@st.cache_data
def run_monte_carlo(data_series, days=252, sims=1000):
    return helper_instance.monte_carlo_gbm(data_series, days, sims)

@st.cache_data
def running_ef(data_series, days=252, method_ef="Basic", target="constant_variance"):
    return helper_instance.run_ef(data_series, days, method_ef,target)

@st.cache_data
def running_bl(data_series, market_prices, mcaps, views, risk_free_rate):
    return helper_instance.get_max_sharpe_bl(data_series, market_prices, mcaps, views, risk_free_rate)


@st.cache_data
def running_hrp(df):
    return helper_instance.get_rec_bipart(df)

col1, col2 = st.columns([1.2, 4])
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Data","Monte Carlo","EF", "Black-Litterman"
                                        , "HRP", "Portofolio", "Stock Analysis", "PPP"])

with st.sidebar:
    st.markdown("### Risk Parameters")
    text_input = st.text_area(
        "Enter Tickers (comma or newline separated)", 
        value="BBRI, BBCA, ADRO, BBNI, ^JKSE",
        height=150
    )
    text_input4 = st.text_input("Enter Historical Years eg. 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max", value="5y")
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        text_input2 = st.text_input("Enter Days", value="252")
    with sub_col2:    
        text_input3 = st.text_input("Enter Simulation", value="1000")
    ticker_list = [item.strip().upper() for item in text_input.replace('\n', ',').split(',') if item.strip()]
    
with col1:
    st.markdown("### Data Parameters")
    capital = st.number_input("Insert a capital (IDR)", value=1_000_000_000)
    risk_free_rate = st.number_input("Enter Risk Free Rate", value=0.06)
    options_data = ["In-Sample", "Full"]
    selection = st.segmented_control(
        "Data", options_data, selection_mode="single",default = "In-Sample"
    )
    p_data = st.slider("IS Split : ", 0.5, 0.9, 0.7)
    

with col2:
    if not ticker_list:
        st.markdown("### Please enter at least one ticker symbol.")
    else:
        df = get_data(ticker_list,text_input4)
        if "^JKSE" in ticker_list:
            IHSG = df["^JKSE"]
            df = df.drop("^JKSE", axis=1)
        else:
            pass
        split_index = int(len(df) * p_data)
        train_df = df.iloc[:split_index].copy()
        test_df  = df.iloc[split_index:].copy()
        data_new = train_df if selection == "In-Sample" else df.copy()
        IHSG_train = IHSG.iloc[:split_index].copy() if selection == "In-Sample" else IHSG.copy()
        with tab2:
            fig_mc = go.Figure()
            option = st.selectbox("Choose Monte Carlo Symbol", (df.columns))
            monte_t = run_monte_carlo(df[option], int(text_input2), int(text_input3))
            total_path = monte_t.shape[1]
            for i in range(total_path): 
                fig_mc.add_trace(go.Scattergl(
                    y=monte_t[:, i], 
                    mode='lines', 
                    line=dict(width=1, color='rgba(0, 191, 255, 0.15)')
                ))

            fig_mc.update_layout(
                showlegend=False, 
                margin=dict(l=0, r=0, t=30, b=0),
                title="Monte Carlo Paths"
            )
            p5 = np.percentile(monte_t, 5, axis=1)
            p50 = np.median(monte_t, axis=1)
            p95 = np.percentile(monte_t, 95, axis=1)
            
            summary_df = pd.DataFrame({
                "5%": p5,
                "Median": p50,
                "95%": p95
            })
            
            st.markdown(f"### Monte Carlo Simulation ({option})")
            st.line_chart(summary_df)
            st.plotly_chart(fig_mc, use_container_width=True, key="mc_chart")
        with tab1:
            st.markdown("### Historical Data")
            st.dataframe(df, use_container_width=True)
            
        with tab3:
            option2 = st.selectbox("Choose Method", ("Basic", "Ledoit Wolf"))
            if option2 == "Ledoit Wolf":
                option_wolf = st.selectbox("Choose Target", ("constant_variance", "single_factor"
                                                             , "constant_correlation"))
            else:
                option_wolf = "constant_variance"
            sub_col3, sub_col4 = st.columns([3, 1])
            fig_ef = go.Figure()
            ef_vols, ef_returns = running_ef(data_new,int(text_input2),method_ef=option2, target=option_wolf)
            ms_vol, ms_return, ms_weights = helper_instance.get_max_sharpe(data_new, int(text_input2), method_ef=option2, risk_free_rate=risk_free_rate, target=option_wolf)
            with sub_col3:
                fig_ef.add_trace(go.Scatter(
                    x=ef_vols, 
                    y=ef_returns, 
                    mode='lines', 
                    line=dict(color='rgba(0, 191, 255, 1)', width=3),
                    name='Optimal Frontier'
                ))
                
                fig_ef.add_trace(go.Scatter(
                    x=[ms_vol], 
                    y=[ms_return], 
                    mode='markers',
                    marker=dict(size=18, color='red', symbol='star'),
                    name='Max Sharpe Portfolio'
                ))
                
                fig_ef.update_layout(
                    title="Optimized Efficient Frontier Curve",
                    xaxis_title="Expected Volatility (Risk)",
                    yaxis_title="Target Return",
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig_ef, use_container_width=True, key="ef_curve_chart")
            with sub_col4:
                st.markdown("### Max Sharpe Asset Allocation")
                weight_df = pd.DataFrame({"Allocation": ms_weights}, index=data_new.columns)
                weight_df = weight_df.sort_values(by='Allocation', ascending=False)
                st.dataframe(weight_df.style.format("{:.2%}"), use_container_width=True)

            st.markdown("### Backtest on allocation")
            if selection == "In-Sample":
                split_date = df.index[split_index]
                
                is_prices = df.iloc[:split_index]
                oos_prices = df.iloc[split_index:]
                
                cum_is = is_prices / is_prices.iloc[0]
                is_values = (cum_is * (ms_weights * capital)).sum(axis=1)
                
                capital_at_oos_start = is_values.iloc[-1]
                
                cum_oos = oos_prices / oos_prices.iloc[0]
                oos_values = (cum_oos * (ms_weights * capital_at_oos_start)).sum(axis=1)
                
                total_value_series = pd.concat([is_values.iloc[:-1], oos_values])
                
                close_backtest = pd.DataFrame(index=total_value_series.index)
                close_backtest["Total_Value"] = total_value_series
                close_backtest["Type"] = np.where(close_backtest.index < split_date, "IS", "OOS")
                
                fig_port = px.line(close_backtest, y="Total_Value", color="Type", template="plotly_dark")
                fig_port.add_vline(x=split_date, line_dash="dot", line_color="yellow")

                st.plotly_chart(fig_port, use_container_width=True, key="ef_backtest_chart")
            else:
                st.markdown("### Use In-Sample for backtest")
        
        with tab4:
            st.markdown("### Black-Litterman Views")
            if "^JKSE" in ticker_list:
                view_df = pd.DataFrame({
                    "Asset": df.columns,
                    "Use_View": [False] * len(df.columns),
                    "Expected_Return": [0.05] * len(df.columns)
                })
                
                edited_df = st.data_editor(
                    view_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Expected_Return": st.column_config.NumberColumn(format="%.2f", step=0.01)
                    }
                )
                
                views = Bl_input_data(edited_df)
                
                if st.button("Run Black-Litterman Analysis"):
                    with st.spinner("Calculating BL Posterior and running backtests..."):
                        m_cap = helper_instance.get_market_caps(train_df)
                        total_mcap = sum(m_cap.values())
                        prior_weights = {ticker: cap / total_mcap for ticker, cap in m_cap.items()}
                        
                        bl_vol, bl_return, bl_weights = running_bl(train_df, IHSG_train, m_cap, views, risk_free_rate)
                        _, _, w_basic = helper_instance.get_max_sharpe(train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic")
                        _, _, w_lw = helper_instance.get_max_sharpe(train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf")
                        weights = helper_instance.get_rec_bipart(train_df)
                        
                        comparison_df = pd.DataFrame({
                            "Market Prior": prior_weights,
                            "BL Posterior": dict(zip(df.columns, bl_weights))
                        }).reset_index().rename(columns={"index": "Asset"})
                        
                        fig_weights = px.bar(
                            comparison_df, 
                            x="Asset", 
                            y=["Market Prior", "BL Posterior"],
                            barmode="group",
                            title="Weight Shift: Market Equilibrium vs. Your Views",
                            template="plotly_dark",
                            color_discrete_map={"Market Prior": "#7F7F7F", "BL Posterior": PRIMARY_COLOR}
                        )
                        st.plotly_chart(fig_weights, use_container_width=True, key="bl_weights_chart")
                        
                        st.markdown("### Strategy Backtest (Forward-Walking PnL)")
                        
                        cum_return_test = test_df / test_df.iloc[0]
                        
                        pnl_df = pd.DataFrame(index=test_df.index)
                        pnl_df["Basic EF (Max Sharpe)"] = (cum_return_test * (w_basic * capital)).sum(axis=1)
                        pnl_df["Ledoit Wolf EF"] = (cum_return_test * (w_lw * capital)).sum(axis=1)
                        pnl_df["Black-Litterman"] = (cum_return_test * (bl_weights * capital)).sum(axis=1)
                        pnl_df["HRP"] = (cum_return_test * (weights * capital)).sum(axis=1)
                        
                        sharep_calt_bl = helper_instance.compute_sharpe(pnl_df, risk_free_rate=risk_free_rate)
                        
                        fig_pnl = px.line(
                            pnl_df, 
                            title="Out-Of-Sample Portfolio Growth (Nominal IDR)", 
                            template="plotly_dark",
                            color_discrete_sequence=["#FF4B4B", "#FFA500", PRIMARY_COLOR, "#00FF00"]
                        )
                        
                        st.plotly_chart(fig_pnl, use_container_width=True, key="bl_pnl_chart")
                        
                        st.markdown("### Optimized Allocation Table & OOS Sharpe")
                        st.dataframe(
                            comparison_df.set_index("Asset").style.format("{:.2%}"), 
                            use_container_width=True
                        )
                        sharpe_df = sharep_calt_bl.to_frame(name="Sharpe Ratio")
                        st.dataframe(sharpe_df.style.format("{:.2f}"))
            
            else:
                st.markdown("### Please Add ^JKSE in the symbol")
        with tab5:
            st.markdown("### HRP")
            daily_returns_symbol = data_new.pct_change().dropna()
            st.markdown("### Asset Clustering Dendrogram")
            data_array = daily_returns_symbol.T.values
            labels = daily_returns_symbol.columns.tolist()
            
            fig = ff.create_dendrogram(
                data_array,
                labels=labels,
                distfun=helper_instance.get_hrp_distance,
                linkagefun=lambda x: linkage(x, method='single'),
                orientation='bottom'
            )
            fig.update_layout(
                height=600, 
                showlegend=False, 
                xaxis_title="Tickers", 
                yaxis_title="Distance"
            )
            st.plotly_chart(fig, use_container_width=True,key="hrp_dendrogram_chart")
            weights = running_hrp(data_new)
            weight_HRP = pd.DataFrame({"Allocation": weights}, index=data_new.columns)
            st.markdown("### Weight Allocation")
            weight_HRP = weight_HRP.sort_values(by="Allocation", ascending=False)
            st.dataframe(weight_HRP.style.format("{:.2%}"), use_container_width=True)
            st.markdown("### Backtest")
            if selection == "In-Sample":
                weight_HRP["Nominal_IDR"] = weight_HRP["Allocation"] * capital
                close_HRP = df.copy()
                cum_return = close_HRP / close_HRP.iloc[0]
                for i in close_HRP.columns:
                    close_HRP[f"Return_{i}"] = cum_return[i] * weight_HRP.loc[i, "Nominal_IDR"]
                value_columns = [col for col in close_HRP.columns if "Return_" in col]
                close_HRP["Total_Value"] = close_HRP[value_columns].sum(axis=1)
                close_HRP["PNL_Pct"] = ((close_HRP["Total_Value"] / capital) - 1) * 100
                split_date = df.index[split_index]
                close_HRP["Type"] = np.where(close_HRP.index < split_date, "IS", "OOS")
                fig_port_HRP = px.line(close_HRP, y="Total_Value", color="Type", template="plotly_dark")
                fig_port_HRP.add_vline(x=split_date, line_dash="dot", line_color="yellow")

                st.plotly_chart(fig_port_HRP, use_container_width=True, key="hrp_backtest_chart")
            else:
                st.markdown("### Use In-Sample for backtest")

        with tab6:
            st.markdown("### Portofolio Analysis")
            m_cap = helper_instance.get_market_caps(train_df)
            total_mcap = sum(m_cap.values())
            prior_weights = {ticker: cap / total_mcap for ticker, cap in m_cap.items()}
            bl_vol, bl_return, bl_weights = running_bl(train_df, IHSG_train, m_cap, views, risk_free_rate)
            _, _, w_basic = helper_instance.get_max_sharpe(train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic")
            _, _, w_lw = helper_instance.get_max_sharpe(train_df, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf")
            weights = helper_instance.get_rec_bipart(train_df)
            cum_return = test_df / test_df.iloc[0]
            pnl_df = pd.DataFrame(index=test_df.index)
            pnl_df["Basic EF"] = (cum_return * (w_basic * capital)).sum(axis=1)
            pnl_df["Ledoit Wolf EF"] = (cum_return * (w_lw * capital)).sum(axis=1)
            pnl_df["Black-Litterman"] = (cum_return * (bl_weights * capital)).sum(axis=1)
            pnl_df["HRP"] = (cum_return * (weights * capital)).sum(axis=1)
            fig_pnl = px.line(
                pnl_df, 
                title="Out-Of-Sample Portfolio Growth (Nominal IDR)", 
                template="plotly_dark",
                color_discrete_sequence=["#FF4B4B", "#FFA500", PRIMARY_COLOR, "#00FF00"]
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True, key="Every_pnl_chart")
            
            sliding = st.number_input("Sliding (Warmup)", value=252)
            step = st.number_input("Rebalance", value=120)
            if st.button("Run Portofolio Backtest (rebalance)"):
                results = []

                portfolio_value = {
                    "Basic EF": capital,
                    "Black-Litterman" : capital,
                    "HRP": capital,
                    "Ledoit Wolf EF": capital,
                    "Equal": capital,
                    "IHSG": capital,
                    "PPP": capital
                }

                for a in range(sliding, len(df) - step, step):
                    data = df.iloc[a - sliding : a]
                    data_ihsg = IHSG.iloc[a - sliding : a]
                    m_cap = helper_instance.get_market_caps(data)
                    total_mcap = sum(m_cap.values())
                    prior_weights = {ticker: cap / total_mcap for ticker, cap in m_cap.items()}
                    _, _, w_basic = helper_instance.get_max_sharpe(data, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Basic")
                    bl_vol, bl_return, bl_weights = helper_instance.get_max_sharpe_bl(data, data_ihsg, m_cap, views, risk_free_rate)
                    _, _, w_lw = helper_instance.get_max_sharpe(data, int(text_input2), risk_free_rate=risk_free_rate, method_ef="Ledoit Wolf")
                    weights_hrp = helper_instance.get_rec_bipart(data)
                    X, returns_custom, clean_cols = custom_instance.data_preparation(data, window=90)
                    theta_opt = custom_instance.optimize_theta(X, returns_custom, gamma=5)
                    weight_custom_matrix = custom_instance.getting_weight(X, theta_opt)
                    w_custom_latest = weight_custom_matrix[-1]
                    w_custom_series = pd.Series(w_custom_latest, index=clean_cols)
                    w_custom_series = w_custom_series / w_custom_series.sum()
                    weights_eq = pd.Series(1 / data.shape[1], index=data.columns)
                    
                    oos_period = df.iloc[a : a + step]
                    oos_ihsg = IHSG.iloc[a : a + step]

                    cum_ret = oos_period / oos_period.iloc[0]
                    cum_ihsg = oos_ihsg / oos_ihsg.iloc[0]
                    common_cols = [c for c in data.columns if c in clean_cols]
                    pnl_window = pd.DataFrame(index=oos_period.index)
                    pnl_window["Black-Litterman"] = (cum_ret * (bl_weights * portfolio_value["Black-Litterman"])).sum(axis=1)
                    pnl_window["Basic EF"] = (cum_ret * (w_basic * portfolio_value["Basic EF"])).sum(axis=1)
                    pnl_window["HRP"]           = (cum_ret * (weights_hrp * portfolio_value["HRP"])).sum(axis=1)
                    pnl_window["Ledoit Wolf EF"]= (cum_ret * (w_lw       * portfolio_value["Ledoit Wolf EF"])).sum(axis=1)
                    pnl_window["Equal"]         = (cum_ret * (weights_eq  * portfolio_value["Equal"])).sum(axis=1)
                    pnl_window["IHSG"]          = cum_ihsg * portfolio_value["IHSG"]
                    pnl_window["PPP"]           = (cum_ret[common_cols] * (w_custom_series * portfolio_value["PPP"])).sum(axis=1)

                    results.append(pnl_window)

                    for col in portfolio_value:
                        portfolio_value[col] = pnl_window[col].iloc[-1]

                final_pnl = pd.concat(results)
                    
                fig_rebalance = px.line(final_pnl, x=final_pnl.index, y=final_pnl.columns)

                st.markdown(f"### Rebalance Backtest")
                st.plotly_chart(fig_rebalance, use_container_width=True, key="Porto_rebalance_backtest")
                
                sharep_calt = helper_instance.compute_sharpe(final_pnl, risk_free_rate=risk_free_rate)
                sharpe_df = sharep_calt.to_frame(name="Sharpe Ratio")
                st.dataframe(sharpe_df.style.format("{:.2f}"))
            
            
            option_mc = st.selectbox("Choose Method", ("Basic EF", "Ledoit Wolf EF", "Black-Litterman"
                                            ,"HRP"))   
            if st.button("Run MC for Portofolio"):
                fig_mc_port = go.Figure()
                monte_t_port = run_monte_carlo(pnl_df[option_mc], int(text_input2), int(text_input3))
                total_path_mc = monte_t_port.shape[1]
                for i in range(total_path_mc): 
                    fig_mc_port.add_trace(go.Scattergl(
                        y=monte_t_port[:, i], 
                        mode='lines', 
                        line=dict(width=1, color='rgba(0, 191, 255, 0.15)')
                    ))

                fig_mc_port.update_layout(
                    showlegend=False, 
                    margin=dict(l=0, r=0, t=30, b=0),
                    title="Monte Carlo Paths"
                )
                statistic_porto = helper_instance.calculated_risk_metric(monte_t_port, pnl_df[option_mc][-1])
                
                st.markdown(f"### Monte Carlo Simulation ({option_mc})")
                st.line_chart(statistic_porto)
                st.plotly_chart(fig_mc_port, use_container_width=True, key="Porto_mc_chart")
                final_sim_values = monte_t_port[-1, :]
                var_threshold = statistic_porto["5%_Val"].iloc[-1]
                cvar_loss = statistic_porto["CVaR_95"].iloc[-1]
                initial_val = pnl_df[option_mc][-1]
                cvar_threshold = initial_val - cvar_loss

                fig_CVAR = px.histogram(x=final_sim_values, nbins=100)
                fig_CVAR.add_vline(x=var_threshold, line_dash="dash", line_color="orange")
                fig_CVAR.add_vline(x=cvar_threshold, line_dash="dash", line_color="red")

                st.plotly_chart(fig_CVAR, key="CVAR_distribution")
                fig_risk_time = px.line(statistic_porto, y=["Median_Val", "5%_Val"])
                cvar_vals = pnl_df[option_mc][-1] - statistic_porto["CVaR_95"]
                fig_risk_time.add_scatter(y=cvar_vals, mode='lines', name='CVaR_95', line=dict(dash='dot', color='red'))

                st.plotly_chart(fig_risk_time, key="CVAR_timeseries")
        
        with tab7:
            st.markdown("### Stock Analysis")
            analysis_data = {}
            for q in df.columns:
                asset_r = df[q].pct_change().dropna()
                market_r = IHSG.pct_change().dropna()
                alpha, idio_vol,ir = helper_instance.idio_vol_alpha(asset_r,market_r,annualize=True, trading_days=int(text_input2))
                analysis_data[q] = {
                    "Alpha": alpha,
                    "Idiosyncratic Volatility": idio_vol,
                    "IR": ir
                }
            idio = pd.DataFrame(analysis_data).T
            st.dataframe(
                idio.style.format({
                    "Alpha": "{:.2%}",
                    "Idiosyncratic Volatility": "{:.2%}",
                    "Information Ratio": "{:.2f}"
                })
            )
        with tab8:
            st.markdown("### PPP")
            X, returns_custom, clean_cols = custom_instance.data_preparation(data_new, window=60)
            theta_opt = custom_instance.optimize_theta(X, returns_custom, gamma=5)
            weight_custom_matrix = custom_instance.getting_weight(X, theta_opt)
            w_custom_latest = weight_custom_matrix[-1]
            w_custom_series = pd.Series(w_custom_latest, index=clean_cols)
            w_custom_series = w_custom_series / w_custom_series.sum()
            weight_PPP = pd.DataFrame({"Allocation": w_custom_series}, index=w_custom_series.index)
            st.dataframe(weight_PPP.style.format("{:.2%}"), use_container_width=True)
            if selection == "In-Sample":
                weight_PPP["Nominal_IDR"] = weight_PPP["Allocation"] * capital
                close_ppp = df.copy()
                cum_return = close_ppp / close_ppp.iloc[0]
                for i in close_ppp.columns:
                    close_ppp[f"Return_{i}"] = cum_return[i] * weight_HRP.loc[i, "Nominal_IDR"]
                value_columns = [col for col in close_ppp.columns if "Return_" in col]
                close_ppp["Total_Value"] = close_ppp[value_columns].sum(axis=1)
                close_ppp["PNL_Pct"] = ((close_ppp["Total_Value"] / capital) - 1) * 100
                split_date = df.index[split_index]
                close_ppp["Type"] = np.where(close_ppp.index < split_date, "IS", "OOS")
                fig_port_HRP = px.line(close_ppp, y="Total_Value", color="Type", template="plotly_dark")
                fig_port_HRP.add_vline(x=split_date, line_dash="dot", line_color="yellow")

                st.plotly_chart(fig_port_HRP, use_container_width=True, key="PPP_backtest_chart")
            else:
                st.markdown("### Use In-Sample for backtest")
            
            
            
                
                
            
             
            

            
            
