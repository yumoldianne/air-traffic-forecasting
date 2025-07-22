library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(zoo)
library(forecast)
library(TSA)
library(astsa)
library(tseries)
library(fGarch)

data = read_csv("air_traffic.csv")

# Create Date column and sort
data = data %>%
  mutate(Date = as.Date(paste(Year, Month, "01", sep = "-"))) %>%
  arrange(Date)

# Create monthly time series object
start_year = year(min(data$Date))
start_month = month(min(data$Date))
pax = ts(data$Pax, start = c(start_year, start_month), frequency = 12)
plot(pax, main = "Monthly Pax Over Time", ylab = "Pax", xlab = "Year")

adf.test(pax) # -> data is not  so we must check if we can perform differencing
Box.test(pax, type = "Ljung") # -> data is autocorrelated

# We now transform the data and difference for stationarity
pax_l = log(pax)
pax_l_d1 = diff(pax_l)
pax_l_d1.12 = diff(pax_l_d1, lag = 12)  # seasonal differencing

# Plot all stages
plot(
  cbind(pax, pax_l, pax_l_d1, pax_l_d1.12),
  col = "red",
  las = 0  # makes y-axis labels horizontal
)


# Stationarity Test
adf.test(pax_l_d1.12) # -> data is stationary

# ACF/PACF for differenced data
acf(pax_l_d1.12, lag.max = 100)
pacf(pax_l_d1.12, lag.max = 100)

# We now fit the SARIMA Models
sarima.arma11 = Arima(pax_l, order = c(1, 1, 1), seasonal = list(order = c(1, 1, 0), period = 12))
sarima.ma1 = Arima(pax_l, order = c(0, 1, 1), seasonal = list(order = c(1, 1, 0), period = 12))
sarima.ar1 = Arima(pax_l, order = c(1, 1, 0), seasonal = list(order = c(1, 1, 0), period = 12))
sarima.arma12 = Arima(pax_l, order = c(1, 1, 2), seasonal = list(order = c(1, 1, 0), period = 12))
sarima.arma13 = Arima(pax_l, order = c(1, 1, 3), seasonal = list(order = c(1, 1, 0), period = 12))

# AIC comparison
aic_values = c(AR1 = sarima.ar1$aic, MA1 = sarima.ma1$aic, ARMA11 = sarima.arma11$aic, ARMA12 = sarima.arma12$aic, ARMA13 = sarima.arma13$aic)
summary(sarima.ar1)
summary(sarima.ma1)
summary(sarima.arma11) # This is the best model in terms of AIC
summary(sarima.arma12)
summary(sarima.arma13)
print(aic_values)

# Forecasting using the SARIMA model
sfor = forecast(sarima.arma11, h=12)
plot(sfor)

# Residual diagnostics for best SARIMA by getting the residuals after fitting SARIMA(1,1,0)
res.sarima.arma11 = residuals(sarima.arma11)

# Check the residuals
acf(res.sarima.arma11)
pacf(res.sarima.arma11)
Box.test(res.sarima.arma11, type ="Ljung")

# Check the squared residuals
acf(res.sarima.arma11^2)
pacf(res.sarima.arma11^2)
Box.test(res.sarima.arma11^2, type ="Ljung")

# First pass: SARIMA model on log-differenced pax data
sarima_model = Arima(pax_l_d1.12, order = c(1,1,1), seasonal = c(1,1,0), include.mean = FALSE)

# Extract residuals
resid_mean = residuals(sarima_model)
print(resid_mean)

# Second pass: Fit different GARCH-type models

# GARCH(1,0) on SARIMA residuals
garch_10 = garchFit(~ garch(1, 0), data = resid_mean, trace = FALSE)

# GARCH(1,1) on SARIMA residuals
garch_11 = garchFit(~ garch(1, 1), data = resid_mean, trace = FALSE)

# ARMA(1,1)-GARCH(1,1) on preprocessed series directly
arma11_garch11 = garchFit(~ arma(1, 1) + garch(1, 1), data = pax_l_d1.12, trace = FALSE)

# ARMA(2,1)-GARCH(1,1) on preprocessed series directly
arma21_garch11 = garchFit(~ arma(2, 1) + garch(1, 1), data = pax_l_d1.12, trace = FALSE)

# ARMA(2,0)-GARCH(1,1) on preprocessed series directly
arma20_garch11 = garchFit(~ arma(2, 0) + garch(1, 1), data = pax_l_d1.12, trace = FALSE)

# === Step 5: Model Comparison ===
cat("=== GARCH(1,0) ===\n"); summary(garch_10)
cat("\n=== GARCH(1,1) ===\n"); summary(garch_11)
cat("\n=== ARMA(1,1) + GARCH(1,1) ===\n"); summary(arma11_garch11)
cat("\n=== ARMA(2,1) + GARCH(1,1) ===\n"); summary(arma21_garch11)
cat("\n=== ARMA(2,0) + GARCH(1,1) ===\n"); summary(arma20_garch11)

# Coefficients and SE for ARMA(2,1)-GARCH(1,1)
print(coef(arma21_garch11))
print(arma21_garch11@fit$se.coef)

# Testing reduced models
# ARMA(1,0)-GARCH(1,1) on preprocessed series directly
arma10_garch11 = garchFit(~ arma(1, 0) + garch(1, 1), data = pax_l_d1.12, trace = FALSE)

# ARMA(0,1)-GARCH(1,1) on preprocessed series directly
arma01_garch11 = garchFit(~ arma(0, 1) + garch(1, 1), data = pax_l_d1.12, trace = FALSE)

cat("=== ARMA(1,0) + GARCH(1,1) ===\n"); summary(arma10_garch11)
cat("=== ARMA(0,1) + GARCH(1,1) ===\n"); summary(arma01_garch11)

# Comparing ARMA(2,0)-GARCH(1,1) and ARMA(2,1)-GARCH(1,1)
resid20 = residuals(arma20_garch11) / volatility(arma20_garch11)

resid21 = residuals(arma21_garch11) / volatility(arma21_garch11)

lags = c(6, 10, 12, 24)
for(l in lags) {
  p20 = Box.test(resid20, lag = l, type="Ljung-Box")$p.value
  p21 = Box.test(resid21, lag = l, type="Ljung-Box")$p.value
  cat("lag",l,
      ": p_ARMA20 =", signif(p20,3),
      ", p_ARMA21 =", signif(p21,3), "\n")
}

# Residual checking for ARMA(2,1) + GARCH(1,1)

# Extract conditional standard deviation (volatility)
fit.vol = volatility(arma21_garch11)

# Calculate standardized residuals: residuals divided by conditional SD
fit.vol.sr = residuals(arma21_garch11) / fit.vol

# Plot conditional volatility
plot(fit.vol, type = "l", main = "Conditional Volatility", ylab = "Volatility")

# Plot standardized residuals
plot(fit.vol.sr, type = "l", main = "Standardized Residuals", ylab = "Value")

# ACF and PACF of standardized residuals
acf(fit.vol.sr, main = "ACF of Standardized Residuals")
pacf(fit.vol.sr, main = "PACF of Standardized Residuals")

# Ljung-Box test for residual autocorrelation (lag 12)
Box.test(fit.vol.sr, lag = 12, type = "Ljung-Box")

# ACF and PACF of squared standardized residuals
acf(fit.vol.sr^2, main = "ACF of Squared Standardized Residuals")
pacf(fit.vol.sr^2, main = "PACF of Squared Standardized Residuals")

# Ljung-Box test for squared residuals (check for remaining ARCH effects)
Box.test(fit.vol.sr^2, lag = 12, type = "Ljung-Box")

# Residual checking for ARMA(1,0) + GARCH(1,1)
fit.vol = volatility(arma10_garch11)
fit.vol.sr = residuals(arma10_garch11) / fit.vol
plot(fit.vol, type = "l", main = "Conditional Volatility", ylab = "Volatility")
plot(fit.vol.sr, type = "l", main = "Standardized Residuals", ylab = "Value")
acf(fit.vol.sr, main = "ACF of Standardized Residuals")
pacf(fit.vol.sr, main = "PACF of Standardized Residuals")
Box.test(fit.vol.sr, lag = 12, type = "Ljung-Box")
acf(fit.vol.sr^2, main = "ACF of Squared Standardized Residuals")
pacf(fit.vol.sr^2, main = "PACF of Squared Standardized Residuals")
Box.test(fit.vol.sr^2, lag = 12, type = "Ljung-Box")

# Residual checking for ARMA(0,1) + GARCH(1,1)
fit.vol = volatility(arma01_garch11)
fit.vol.sr = residuals(arma01_garch11) / fit.vol
plot(fit.vol, type = "l", main = "Conditional Volatility", ylab = "Volatility")
plot(fit.vol.sr, type = "l", main = "Standardized Residuals", ylab = "Value")
acf(fit.vol.sr, main = "ACF of Standardized Residuals")
pacf(fit.vol.sr, main = "PACF of Standardized Residuals")
Box.test(fit.vol.sr, lag = 12, type = "Ljung-Box")
acf(fit.vol.sr^2, main = "ACF of Squared Standardized Residuals")
pacf(fit.vol.sr^2, main = "PACF of Squared Standardized Residuals")
Box.test(fit.vol.sr^2, lag = 12, type = "Ljung-Box")

# Residual checking for ARMA(1,1) + GARCH(1,1)
fit.vol = volatility(arma11_garch11)
fit.vol.sr = residuals(arma11_garch11) / fit.vol
plot(fit.vol, type = "l", main = "Conditional Volatility", ylab = "Volatility")
plot(fit.vol.sr, type = "l", main = "Standardized Residuals", ylab = "Value")
acf(fit.vol.sr, main = "ACF of Standardized Residuals")
pacf(fit.vol.sr, main = "PACF of Standardized Residuals")
Box.test(fit.vol.sr, lag = 12, type = "Ljung-Box")
acf(fit.vol.sr^2, main = "ACF of Squared Standardized Residuals")
pacf(fit.vol.sr^2, main = "PACF of Squared Standardized Residuals")
Box.test(fit.vol.sr^2, lag = 12, type = "Ljung-Box")

# Residual checking for ARMA(2,0) + GARCH(1,1)
fit.vol = volatility(arma20_garch11)
fit.vol.sr = residuals(arma20_garch11) / fit.vol
plot(fit.vol, type = "l", main = "Conditional Volatility", ylab = "Volatility")
plot(fit.vol.sr, type = "l", main = "Standardized Residuals", ylab = "Value")
acf(fit.vol.sr, main = "ACF of Standardized Residuals")
pacf(fit.vol.sr, main = "PACF of Standardized Residuals")
Box.test(fit.vol.sr, lag = 12, type = "Ljung-Box")
acf(fit.vol.sr^2, main = "ACF of Squared Standardized Residuals")
pacf(fit.vol.sr^2, main = "PACF of Squared Standardized Residuals")
Box.test(fit.vol.sr^2, lag = 12, type = "Ljung-Box")

arma_garch_forecast = predict(arma21_garch11, n.ahead=30,plot=T)
arma_garch_forecast