import pandas as pd
import numpy as np
import yfinance as yf
import logging
import scipy.optimize as sco
from pypfopt import risk_models,black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
import statsmodels.api as sm

LOG_FILE    = "error_download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

class HELPER():
    def __init__(self) -> None:
        pass
    def data_downloader(self, tickers_raw:list[str], period:str, interval:str):
        """
        Data Downloader from yfinance
        
        args:
            tickers_raw (list) : list of tickers
            period (str) : 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            interval (str) : 1 day
        
        returns:
        Dataframe
        """
        try:
            tickers = [c if c == "^JKSE" else f"{c}.JK" for c in tickers_raw]
            df = yf.download(
            tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=True,
            )
            return df
        except Exception as exc:
            log.error(f"Batch download failed: {exc}")
            return pd.DataFrame()
        
    def get_close_data(self, df):
        data = df["Close"]
        return data
    
    def monte_carlo_gbm(self, df, days, path):
        """
        simulates futures price using geometric brownian motion
        
        args:
            df (Dataframe) : historical data
            days (int) : number of futures days to simulate
            path (str) : how many simulation path
        
        returns:
        np.ndarray
        """
        
        dt = 1/days
        df = df.dropna()
        s0 = df.iloc[-1]
        Mu = df.pct_change().mean() *252
        sigma = df.pct_change().std() * np.sqrt(252)
        prices = np.zeros((days+1, path))
        prices[0] = s0
        
        for i in range(1, 1+days):
            Z = np.random.standard_normal(path)
            drift = (Mu - ((sigma **2) / 2))*dt
            shock = sigma * np.sqrt(dt)*Z
            prices[i] = prices[i-1] * np.exp(drift+shock)
            
        return prices
    
    def get_portofolio_information(self, weight, returns, cov_matrix, annualized=False):
        """
        Calculates the annualized return and volatility of a portfolio.
        
        Args:
            weight (np.ndarray): Asset weights in the portfolio.
            returns (pd.Series or np.ndarray): Average daily returns of the assets.
            cov_matrix (pd.DataFrame or np.ndarray): Covariance matrix of asset returns.
            
        Returns:
            tuple: (annualized_return, annualized_volatility)
        """
        scale = 1 if annualized else 252
        port_return = np.sum(returns * weight) * scale
        port_volatility = np.sqrt(np.dot(weight.T, np.dot(cov_matrix * scale, weight)))
        return port_return, port_volatility
    
    def minimize_vol(self, weight, returns, cov_matrix, annualized=False):
        """
        Objective function for optimization to minimize volatility.
        
        Args:
            weight (np.ndarray): Asset weights.
            returns (pd.Series or np.ndarray): Average daily returns.
            cov_matrix (pd.DataFrame or np.ndarray): Covariance matrix.
            
        Returns:
            float: Annualized portfolio volatility.
        """
        return self.get_portofolio_information(weight, returns, cov_matrix, annualized=annualized)[1]
    
    def get_efficient_frontier(self, returns, cov_matrix, return_range, annualized=False):
        """
        Calculates the minimum volatility for a given range of target returns.
        
        Args:
            returns (pd.Series or np.ndarray): Expected returns of the assets.
            cov_matrix (pd.DataFrame or np.ndarray): Covariance matrix of the assets.
            return_range (iterable): Array of target returns to optimize against.
            
        Returns:
            list: Efficient frontier volatilities corresponding to the return_range.
        """
        eff_vols = []
        num_assets = len(returns)
        initial_guess = num_assets * [1. / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))

        for target_return in return_range:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.get_portofolio_information(x, returns, cov_matrix, annualized=annualized)[0] - target_return}
            ]
            
            result = sco.minimize(
                self.minimize_vol, 
                initial_guess, 
                args=(returns, cov_matrix, annualized),
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            eff_vols.append(result['fun'])
            if result.success:
                initial_guess = result.x
            
        return eff_vols
    
    def minimize_sharpe(self, weight, returns, cov_matrix, risk_free_rate=0, annualized=False):
        """
        Objective function to minimize the negative Sharpe ratio.
        
        Args:
            weight (np.ndarray): Asset weights.
            returns (pd.Series or np.ndarray): Average daily returns.
            cov_matrix (pd.DataFrame or np.ndarray): Covariance matrix.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.
            
        Returns:
            float: Negative Sharpe ratio.
        """
        port_return, port_vol = self.get_portofolio_information(weight, returns, cov_matrix, annualized=annualized)
        return -(port_return - risk_free_rate) / port_vol
    
    def get_max_sharpe(self, df, days, risk_free_rate=0, method_ef="Basic", target="constant_variance"):
        """
        Finds the portfolio weights that maximize the Sharpe ratio.
        
        Args:
            df (pd.DataFrame): Historical price data.
            days (int): Annualization factor (e.g., 252).
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.
            method_ef (str, optional): Covariance method ("Basic" or "Ledoit Wolf"). Defaults to "Basic".
            target (str, optional): Target for shrinkage. Defaults to "constant_variance".
            
        Returns:
            tuple: (optimal_volatility, optimal_return, optimal_weights)
        """
        daily_return = df.pct_change().dropna()
        returns = daily_return.mean()
        
        if method_ef == "Ledoit Wolf":
            cov_matrix = self.cov_shrinkage(df, target)
            is_annualized = True
            returns = returns * 252
        else:
            cov_matrix = daily_return.cov()
            is_annualized = False
        
        num_assets = len(returns)
        initial_guess = num_assets * [1. / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = sco.minimize(
            self.minimize_sharpe,
            initial_guess,
            args=(returns, cov_matrix, risk_free_rate, is_annualized),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        opt_return, opt_vol = self.get_portofolio_information(result.x, returns, cov_matrix, annualized=is_annualized)
        weights_series = pd.Series(result.x, index=returns.index)
        return opt_vol, opt_return, weights_series
    
    def run_ef(self, df, days, method_ef="Basic", target="constant_variance"):
        """
        Generates the efficient frontier points.
        
        Args:
            df (pd.DataFrame): Historical price data.
            days (int): Annualization factor.
            method_ef (str, optional): Covariance method. Defaults to "Basic".
            target (str, optional): Target for shrinkage. Defaults to "constant_variance".
            
        Returns:
            tuple: (efficient_volatilities, target_returns)
        """
        daily_return = df.pct_change().dropna()
        mean_return = daily_return.mean()
        
        if method_ef == "Ledoit Wolf":
            cov_matrix = self.cov_shrinkage(df, target)
            is_annualized = True
            mean_return = mean_return * days
        else:
            cov_matrix = daily_return.cov()
            is_annualized = False
            
        target_return = np.linspace(
            mean_return.min() * (1 if is_annualized else days), 
            mean_return.max() * (1 if is_annualized else days), 
            50
        )
        efficient_vols = self.get_efficient_frontier(mean_return, cov_matrix, target_return, annualized=is_annualized)
        return efficient_vols, target_return
    
    def cov_shrinkage(self, df:pd.DataFrame, target="constant_variance"):
        """
        Calculates the Ledoit-Wolf shrinkage covariance matrix.
        
        Args:
            df (pd.DataFrame): Historical price data.
            target (str, optional): Shrinkage target. Defaults to "constant_variance".
            
        Returns:
            pd.DataFrame: Shrunk covariance matrix.
        """
        return risk_models.CovarianceShrinkage(df).ledoit_wolf(target)
    
    def get_bl_posterior(self, df:pd.DataFrame, market_prices:pd.Series, mcaps:dict, views:dict,target="constant_variance"):
        """
        Calculates Black-Litterman posterior returns and covariance.
        
        Args:
            df (pd.DataFrame): Historical price data.
            market_prices (pd.Series): Market index prices.
            mcaps (dict): Market capitalizations of the assets.
            views (dict): Absolute views on expected returns.
            target (str, optional): Shrinkage target. Defaults to "constant_variance".
            
        Returns:
            tuple: (bl_posterior_returns, bl_posterior_covariance)
        """
        cov_matrix = self.cov_shrinkage(df,target)
        aligned_mcaps = {k: mcaps[k] for k in cov_matrix.index if k in mcaps}
        delta = black_litterman.market_implied_risk_aversion(market_prices)
        prior_returns = black_litterman.market_implied_prior_returns(aligned_mcaps, delta, cov_matrix)
        
        bl = BlackLittermanModel(cov_matrix, pi=prior_returns, absolute_views=views)
        
        return bl.bl_returns(), bl.bl_cov()

    def get_max_sharpe_bl(self, df, market_prices, mcaps, views, risk_free_rate=0):
        """
        Finds the maximum Sharpe ratio portfolio using the Black-Litterman model.
        
        Args:
            df (pd.DataFrame): Historical price data.
            market_prices (pd.Series): Market index prices.
            mcaps (dict): Market capitalizations.
            views (dict): Absolute views.
            risk_free_rate (float, optional): Risk-free rate. Defaults to 0.
            
        Returns:
            tuple: (optimal_volatility, optimal_return, optimal_weights)
        """
        bl_returns_ann, bl_cov_ann = self.get_bl_posterior(df, market_prices, mcaps, views)
    
        num_assets = len(bl_returns_ann)
        initial_guess = num_assets * [1. / num_assets]
        bounds = tuple((0, 1) for _ in range(num_assets))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        result = sco.minimize(
            self.minimize_sharpe,
            initial_guess,
            args=(bl_returns_ann, bl_cov_ann, risk_free_rate, True),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        opt_return, opt_vol = self.get_portofolio_information(result.x, bl_returns_ann, bl_cov_ann, annualized=True)
        weights_series = pd.Series(result.x, index=bl_returns_ann.index)
        return opt_vol, opt_return, weights_series
        
    def get_cluster_var(self, cov, cluster_index):
        """
        Calculates the variance of a specific cluster for Hierarchical Risk Parity (HRP).
        
        Args:
            cov (pd.DataFrame): Ordered covariance matrix.
            cluster_index (list): Indices of the cluster.
            
        Returns:
            float: Cluster variance.
        """
        c_cov = cov.loc[cluster_index, cluster_index]
        
        diag = np.diag(c_cov)
        diag = np.where(diag <= 1e-10, 1e-10, diag)
        
        ivw = 1.0 / diag
        ivw /= ivw.sum()
        
        clust_var = np.dot(ivw, np.dot(c_cov, ivw))
        return clust_var
    
    def get_rec_bipart(self, df):
        """
        Computes asset weights using Recursive Bisection for HRP.
        
        Args:
            df (pd.DataFrame): Historical price data.
            
        Returns:
            pd.Series: HRP optimized portfolio weights.
        """
        daily_returns_symbol = df.pct_change().dropna()
        corr_matrix = daily_returns_symbol.corr().values 
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0) 
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        D = np.sqrt(0.5 * (1 - corr_matrix))
        D = (D + D.T) / 2.0  
        np.fill_diagonal(D, 0.0) 
        Z = linkage(squareform(D), method="single")
        sort_indices = leaves_list(Z)
        ordered_tickers = daily_returns_symbol.columns[sort_indices]
        ordered_cov = daily_returns_symbol[ordered_tickers].cov()
        w = pd.Series(1.0, index=ordered_tickers)
        c_items = [ordered_tickers]
        
        while len(c_items) > 0:
            c_items_next = []
            
            for items in c_items:
                if len(items) <= 1:
                    continue
                    
                mid = int(len(items) / 2)
                c_left = items[:mid]
                c_right = items[mid:]
                
                v_left = self.get_cluster_var(ordered_cov, c_left)
                v_right = self.get_cluster_var(ordered_cov, c_right)
                
                alpha = 1 - (v_left / (v_left + v_right))
                
                w[c_left] *= alpha
                w[c_right] *= (1 - alpha)
                
                c_items_next.extend([c_left, c_right])
                
            c_items = c_items_next
            
        return w
    
    def calculated_risk_metric(self, monte, initial_value):
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR) from simulations.
        
        Args:
            monte (np.ndarray): Monte Carlo simulation price paths.
            initial_value (float or np.ndarray): Initial portfolio or asset value.
            
        Returns:
            pd.DataFrame: Summary of risk metrics including 5%, median, 95%, VaR_95, and CVaR_95.
        """
        p5 = np.percentile(monte, 5, axis=1)
        p50 = np.percentile(monte, 50, axis=1)
        p95 = np.percentile(monte, 95, axis=1)
        
        var_95 = initial_value - p5
        
        cvar_95 = np.array([
        initial_value - monte[i][monte[i] <= p5[i]].mean() for i in range(len(p5))])
        
        summary_df = pd.DataFrame({
            "5%_Val": p5,
            "Median_Val": p50,
            "95%_Val": p95,
            "VaR_95": var_95,
            "CVaR_95": cvar_95
        })
        
        return summary_df
    
    def get_hrp_distance(self, x):
        """
        Calculates the distance matrix for HRP clustering based on correlation.
        
        Args:
            x (np.ndarray or pd.DataFrame): Input data to calculate correlation from.
            
        Returns:
            np.ndarray: Condensed distance matrix.
        """
        corr_matrix = np.corrcoef(x)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        dist_matrix = (dist_matrix + dist_matrix.T) / 2.0
        np.fill_diagonal(dist_matrix, 0.0)
        return squareform(dist_matrix)
        
    def get_market_caps(self, df):
        """
        Retrieves market capitalizations for given tickers using yfinance.
        
        Args:
            df (pd.DataFrame): DataFrame with tickers as columns.
            
        Returns:
            dict: Dictionary mapping tickers to their calculated or retrieved market caps.
        """
        mcaps = {}
        for c in df.columns:
            tk = yf.Ticker(c)
            
            shares = tk.info.get("sharesOutstanding")
            last_price = df[c].iloc[-1] 
            
            if shares and last_price:
                mcaps[c] = shares * last_price
            else:
                mcaps[c] = tk.info.get("marketCap")
                log.warning(f"Could not get market cap for {c}, using total Mcaps {mcaps[c]}")
        
        return mcaps              
        
    def idio_vol_alpha(self, asset_return, market_return, annualize = True, trading_days = 252) -> float:
        X = sm.add_constant(market_return)
        model = sm.OLS(asset_return, X).fit()
        
        vol = np.sqrt(np.var(model.resid, ddof=1))
        alpha = model.params.iloc[0]
        if annualize:
            alpha = alpha * trading_days
            idio_vol = vol * np.sqrt(trading_days)
            
        ir = alpha/idio_vol if idio_vol != 0 else 0
        
        return alpha, idio_vol, ir
    
    def compute_sharpe(self, pnl_df, risk_free_rate = 0.0) -> pd.Series:
        daily_returns = pnl_df.pct_change().dropna()
        annual_return = daily_returns.mean() * 252
        annual_vol    = daily_returns.std() * np.sqrt(252)
        return (annual_return - risk_free_rate) / annual_vol
    