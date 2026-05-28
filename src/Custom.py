import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pymc as pm
import ta


"""
PPP using yfinance, for any proffesional who use blomberg or other data set
i recomend using those data as yfinance doest have complete data specially 
for fundamental data PBV, PER, etc
"""


class Custom_optimize():
    def __init__(self):
        pass
    def stadarize(self,df):
        return df.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    def momentum(self,df, window):
        log_momentum = np.log(df) - np.log(df.shift(window))
        mom_std = self.stadarize(log_momentum)
        return mom_std
    
    def sma_rolling(self, df, window):
        sma_keep = {}
        for i in df.columns:
            sma_keep[i] = df[i].rolling(window).mean()
        sma_df = pd.DataFrame(sma_keep)
        return self.stadarize(sma_df)
    
    def market_cap_t(self, df):
        mcap = {}
        exclude = []
        for c in df.columns:
            cap = yf.Ticker(c)
            shares = cap.info.get("sharesOutstanding")
            if shares:
                mcap[c] = shares * df[c]
            else:
                exclude.append(c)
        mcap_df = pd.DataFrame(mcap)
        mcap_std = self.stadarize(np.log(mcap_df))
        return mcap_std, exclude
    
    def rsi_data(self, df, window):
        rsi_keep = {}
        for i in df.columns:
            rsi_keep[i] = ta.momentum.RSIIndicator(df[i],window).rsi()
        rsi = pd.DataFrame(rsi_keep)
        rsi_std = self.stadarize(rsi)
        return rsi_std
        
                
    def data_preparation(self, df, window=252):
        mom_std = self.momentum(df, window)
        rsi_std = self.rsi_data(df,window)
        sma_std = self.sma_rolling(df, window)
        mcap_std, exclude = self.market_cap_t(df)
        
        clean_data = df.drop(columns=exclude)
        mom_std = mom_std[clean_data.columns]
        rsi_std = rsi_std[clean_data.columns]
        sma_std = sma_std[clean_data.columns]
        returns_shifted = clean_data.pct_change().shift(-1)
        
        valid_mask = mom_std.notna().all(axis=1) & mcap_std.notna().all(axis=1) & returns_shifted.notna().all(axis=1) & rsi_std.notna().all(axis=1)
        
        X = np.stack([
            mom_std.loc[valid_mask].values,
            mcap_std.loc[valid_mask].values,
            rsi_std.loc[valid_mask].values,
        ], axis=2)
        
        returns_t_plus_1 = returns_shifted.loc[valid_mask].values
        
        return X, returns_t_plus_1, clean_data.columns
                
    def optimize_theta(self, X, returns, gamma=5):
        T, N, K = X.shape
        
        def objective(theta):
            tilt = (X @ theta)/N
            # Benchmark 1/N
            weight = (1/N) + tilt
            weight = np.clip(weight, 0, None)
            weight = weight / weight.sum(axis=1, keepdims=True)
            port_returns = np.sum(weight * returns, axis=1)
            ulitlity = np.mean(port_returns) - (gamma/2) * np.var(port_returns, ddof=1)
            # ulitlity = np.mean((1 + port_returns) ** (1 - gamma) / (1 - gamma))
            return -ulitlity
        initial_theta = np.zeros(K)
        res = minimize(objective,initial_theta,method='L-BFGS-B')
        return res.x
    
    def bayesian_ppp(self, X, returns, gamma=5):
        T, N, K = X.shape
        with pm.Model() as model:
            theta = pm.Normal("theta", mu=0, sigma=1, shape= K)
            tilt = pm.math.dot(X, theta)/N
            raw_weight = (1/N) + tilt
            clip_weight = pm.math.clip(raw_weight, 0, None)
            weight = clip_weight / pm.math.sum(clip_weight,axis=1, keepdims=True)
            port_returns = pm.math.sum(weight * returns, axis=1)
            ulitlity = pm.math.mean(port_returns) - (gamma/2) * pm.math.var(port_returns)
            pm.Potential('utility_obs', ulitlity)
            trace = pm.sample(1000, tune=1000)
        return trace
        
    def getting_BPPP_weight(self, X, trace):
        T, N, K = X.shape
        theta_sample = trace.posterior["theta"].values.reshape(-1,K)
        optimal_theta = np.mean(theta_sample, axis=0)
        N = X.shape[1]
        current_features = X[-1]
        tilt = (current_features @ optimal_theta) / N
        weights = (1 / N) + tilt
        weights = np.clip(weights, 0, None)
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            final_weights = weights / weight_sum
        else:
            final_weights = np.ones(N)/N
        return final_weights    
    
    def getting_weight(self, X, theta_opt):
        T, N, K = X.shape
        tilt = (X @ theta_opt) / N
        weights = (1 / N) + tilt
        weights = np.clip(weights, 0, None)       
        weights = weights / weights.sum(axis=1, keepdims=True)  
        return weights