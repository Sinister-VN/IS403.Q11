import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv('../01_Data/02_Processed/ASEAN_FDI_GDP_Data_Final.csv')

print("="*80)
print("ANALYSIS RESULTS FOR LATEX PAPER")
print("="*80)

# 1. DESCRIPTIVE STATISTICS
print("\n1. DESCRIPTIVE STATISTICS")
print("-"*80)
print(f"Total observations: {len(df)}")
print(f"Number of countries: {len(df['Country'].unique())}")
print(f"Countries: {', '.join(sorted(df['Country'].unique()))}")
print(f"Year range: {df['Year'].min()}-{df['Year'].max()}")

print("\nOverall statistics:")
desc = df[['FDI', 'GDP_Growth', 'Trade_Openness']].describe()
print(desc)

print("\n2. FDI BY COUNTRY (RANKING)")
print("-"*80)
avg_fdi_by_country = df.groupby('Country')['FDI'].mean().sort_values(ascending=False)
print("\nAverage FDI (million USD):")
for rank, (country, fdi) in enumerate(avg_fdi_by_country.items(), 1):
    print(f"  {rank:2d}. {country:25s}: {fdi/1e9:8.2f} billion USD")

# Vietnam stats
vnm = df[df['Country'] == 'Viet Nam']
print(f"\nVietnam statistics:")
print(f"  Average FDI: {vnm['FDI'].mean()/1e9:.2f} billion USD")
print(f"  Average GDP Growth: {vnm['GDP_Growth'].mean():.2f}%")
print(f"  Average Trade Openness: {vnm['Trade_Openness'].mean():.2f}%")

# 3. CORRELATION ANALYSIS
print("\n3. CORRELATION MATRIX")
print("-"*80)
# Remove NaN
df_clean = df[['FDI', 'GDP_Growth', 'Trade_Openness']].dropna()
corr_matrix = df_clean.corr()
print(corr_matrix)

# Individual correlations
corr_fdi_gdp, p_fdi_gdp = pearsonr(df_clean['FDI'], df_clean['GDP_Growth'])
corr_fdi_trade, p_fdi_trade = pearsonr(df_clean['FDI'], df_clean['Trade_Openness'])
corr_gdp_trade, p_gdp_trade = pearsonr(df_clean['GDP_Growth'], df_clean['Trade_Openness'])

print(f"\nFDI ↔ GDP_Growth: r = {corr_fdi_gdp:.3f} (p = {p_fdi_gdp:.4f})")
print(f"FDI ↔ Trade_Openness: r = {corr_fdi_trade:.3f} (p = {p_fdi_trade:.4f})")
print(f"GDP_Growth ↔ Trade_Openness: r = {corr_gdp_trade:.3f} (p = {p_gdp_trade:.4f})")

# 4. PANEL REGRESSION (POOLED OLS)
print("\n4. PANEL REGRESSION (POOLED OLS)")
print("-"*80)

# Prepare data
X = df[['GDP_Growth', 'Trade_Openness']]
y = df['FDI']

# Drop NaN
mask = X.notna().all(axis=1) & y.notna()
X_clean = X[mask]
y_clean = y[mask]

print(f"Valid observations: {len(X_clean)} (dropped {len(X) - len(X_clean)} NaN)")

# Fit model
model = LinearRegression()
model.fit(X_clean, y_clean)

r2 = model.score(X_clean, y_clean)
coef_gdp = model.coef_[0]
coef_trade = model.coef_[1]
intercept = model.intercept_

print(f"\nPooled OLS Results:")
print(f"  R-squared: {r2:.4f}")
print(f"  Intercept: {intercept:,.2f}")
print(f"  Coefficient GDP_Growth: {coef_gdp:,.2f}")
print(f"  Coefficient Trade_Openness: {coef_trade:,.2f}")

print(f"\nInterpretation:")
print(f"  1% increase in GDP Growth → FDI increases by {coef_gdp/1e9:.3f} billion USD")
print(f"  1% increase in Trade Openness → FDI increases by {coef_trade/1e9:.3f} billion USD")

# 5. VIETNAM FORECASTING COMPARISON
print("\n5. VIETNAM FORECASTING (2020-2024)")
print("-"*80)
vnm = df[df['Country'] == 'Viet Nam'].sort_values('Year')
train = vnm[vnm['Year'] <= 2019]
test = vnm[vnm['Year'] > 2019]

print(f"Train: {len(train)} obs (2000-2019)")
print(f"Test: {len(test)} obs (2020-2024)")
print(f"\nTest period actual FDI:")
for _, row in test.iterrows():
    print(f"  {int(row['Year'])}: {row['FDI']/1e9:.2f} billion USD")

# Random Walk
from sklearn.metrics import mean_squared_error
forecast_rw = [train['FDI'].iloc[-1]] * len(test)
rmsfe_rw = np.sqrt(mean_squared_error(test['FDI'], forecast_rw))
mape_rw = np.mean(np.abs((test['FDI'] - forecast_rw) / test['FDI'])) * 100

print(f"\nRandom Walk RMSFE: {rmsfe_rw/1e9:.3f} billion USD, MAPE: {mape_rw:.2f}%")

# ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    model_arima = ARIMA(train['FDI'], order=(1,1,1))
    results_arima = model_arima.fit()
    forecast_arima = results_arima.forecast(steps=len(test))
    
    rmsfe_arima = np.sqrt(mean_squared_error(test['FDI'], forecast_arima))
    mape_arima = np.mean(np.abs((test['FDI'] - forecast_arima) / test['FDI'])) * 100
    
    print(f"ARIMA(1,1,1) RMSFE: {rmsfe_arima/1e9:.3f} billion USD, MAPE: {mape_arima:.2f}%")
    
    if rmsfe_arima < rmsfe_rw:
        print(f"\n✅ Best Model: ARIMA (RMSFE: {rmsfe_arima/1e9:.3f}B, MAPE: {mape_arima:.2f}%)")
    else:
        print(f"\n✅ Best Model: Random Walk (RMSFE: {rmsfe_rw/1e9:.3f}B, MAPE: {mape_rw:.2f}%)")
except Exception as e:
    print(f"ARIMA error: {e}")
    print(f"\n✅ Best Model: Random Walk (RMSFE: {rmsfe_rw/1e9:.3f}B, MAPE: {mape_rw:.2f}%)")

print("\n" + "="*80)
print("COMPLETE! Copy these values to LaTeX paper")
print("="*80)
