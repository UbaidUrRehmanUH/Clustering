
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("country_population.csv")



# Select columns with population data
years = list(map(str, range(1960, 2016)))
pop_data = df[years]

# Fill missing values with mean value
pop_data = pop_data.fillna(pop_data.mean())

# Normalize the data
scaler = StandardScaler()
pop_data_norm = scaler.fit_transform(pop_data)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(pop_data_norm)
clusters = kmeans.predict(pop_data_norm)

# Add cluster labels to the dataset
df['Cluster'] = clusters

# Output the results to a new CSV file
df.to_csv('country_population_clusters.csv', index=False)

# Plot the clusters
plt.scatter(df['1960'], df['2016'], c=df['Cluster'])
plt.xlabel('1960 Population')
plt.ylabel('2018 Population')
plt.title('Country Clusters by Population')
plt.show()


from scipy.optimize import curve_fit
from scipy import stats

# Define the err_ranges function to compute confidence intervals
def err_ranges(residuals, alpha):
    n = len(residuals)
    df = n - 2  # degrees of freedom
    t = stats.t(df).ppf(1 - alpha/2)
    std_err = np.sqrt(np.sum(residuals**2) / df)
    err_lower = t * std_err * np.sqrt(1 + 1/n + (x_pred - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    err_upper = t * std_err * np.sqrt(1 + 1/n + (x_pred - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    return err_lower, err_upper


# Load the dataset
df = pd.read_csv("country_population.csv")
df.head()

from scipy.optimize import curve_fit

# Load the dataset
df = pd.read_csv("country_population.csv")

df = df[df['Country Name'] == 'United Kingdom']

# Extract the year and population columns
years = np.array(df.columns[4:], dtype=int)
populations = np.array(df.iloc[:, 4:].values.flatten(), dtype=float)

# Define the model function
def model_func(x, a, b, c):
    return a + b * x + c * x**2

# Fit the population data to the model
params, cov = curve_fit(model_func, years, populations)

# Extract the estimated parameter values and the covariance matrix
a_fit, b_fit, c_fit = params
cov_b_c = cov[1:, 1:]

# Make predictions using the fitted model
years_pred = np.arange(1960, 2050, 1)
populations_pred = model_func(years_pred, a_fit, b_fit, c_fit)

# Calculate the confidence intervals for the parameter estimates
perr = np.sqrt(np.diag(cov))
a_err, b_err, c_err = perr
conf_int_b = (b_fit - 1.96 * b_err, b_fit + 1.96 * b_err)
conf_int_c = (c_fit - 1.96 * c_err, c_fit + 1.96 * c_err)

# Print the parameter estimates and confidence intervals
print(f'a = {a_fit:.2f}')
print(f'b = {b_fit:.2f} ({conf_int_b[0]:.2f}, {conf_int_b[1]:.2f})')
print(f'c = {c_fit:.2f} ({conf_int_c[0]:.2f}, {conf_int_c[1]:.2f})')

# Plot the data and the fitted function with confidence intervals
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(years, populations, label='data')
ax.plot(years_pred, populations_pred, label='fit', color='red')
ax.fill_between(years_pred, model_func(years_pred, a_fit, conf_int_b[0], conf_int_c[0]), 
                model_func(years_pred, a_fit, conf_int_b[1], conf_int_c[1]), alpha=0.2, color='gray')
ax.legend()
plt.show()

