#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from astropy.io.votable import parse
from astropy.constants import c
from tqdm import tqdm
from scipy.optimize import minimize, curve_fit
import numpy as np
import math


# In[2]:


# Read the table file
table_file = r'C:\Users\aadis\Desktop\SED\vizier_votable.vot'
votable = parse(table_file)
table = votable.get_first_table().to_table(use_names_over_ids=True)
tablepd = table.to_pandas()
tablepd['_sed_eflux'] = tablepd['_sed_eflux'].fillna(0)
tablepd['index'] = tablepd.index

# Cleaning up the data for inaccurate catalogues
tablepd = tablepd[~tablepd['_sed_filter'].str.contains('SkyMapper')]
tablepd = tablepd[~tablepd['_sed_filter'].str.contains('XMM')]
tablepd = tablepd[~tablepd['_sed_filter'].str.contains('GAIA')]
tablepd = tablepd[~tablepd['_sed_filter'].str.contains('Gaia')]
tablepd = tablepd[~tablepd['_sed_filter'].str.contains('VISTA')]

# Drop useless columns
tablepd.drop(['_tabname', '_ID'], axis=1, inplace=True)

# Create wavelength column
speed_of_light = 299792458  # meters per second
tablepd['_sed_wavelen[nm]'] = speed_of_light / tablepd['_sed_freq']
tablepd['_sed_flux [W/(m^2 Hz)]'] = tablepd['_sed_flux'] * 10 ** -26
tablepd.drop(['_sed_flux'], axis=1, inplace=True)
tablepd['_sed_eflux [W/(m^2 Hz)]'] = tablepd['_sed_eflux'] * 10 ** -26
tablepd.drop(['_sed_eflux'], axis=1, inplace=True)
tablepd['_sed_freq [Hz]'] = tablepd['_sed_freq'] * 10 ** 9
tablepd.drop(['_sed_freq'], axis=1, inplace=True)

print(len(tablepd))


# In[3]:


# Plot observed data
fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.errorbar(tablepd['_sed_wavelen[nm]'], tablepd['_sed_flux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']),
             yerr=tablepd['_sed_eflux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']), fmt='.', color='green', label='removed')
ax1.set(xscale='log', yscale='log', xlim=(200, 60000), xlabel='Wavelength[nm]',
        ylabel='Flux ($\\nu\ F_{\\nu}$) [$W\ m^{ -2}$]')

# Drop data with too big error bars
for i in tablepd.index:
    if tablepd.loc[i, '_sed_eflux [W/(m^2 Hz)]'] / tablepd.loc[i, '_sed_flux [W/(m^2 Hz)]'] > 0.1:
        tablepd.drop(tablepd.loc[tablepd['index'] == i].index[0], inplace=True)

# Get rid of data with error equals to 0
tablepd = tablepd[tablepd['_sed_eflux [W/(m^2 Hz)]'] != 0]
tablepd.reset_index(inplace=True)

# Get rid of the outliers
index_toBeRemoved = []
rel_error_percentage = 0.2
tablepd.sort_values(by='_sed_wavelen[nm]', inplace=True)
tablepd.reset_index(drop=True, inplace=True)
for i in tablepd.index[1:-1]:
    if (tablepd.loc[i + 1, '_sed_wavelen[nm]'] == tablepd.loc[i, '_sed_wavelen[nm]']) or (
            tablepd.loc[i - 1, '_sed_wavelen[nm]'] == tablepd.loc[i, '_sed_wavelen[nm]']):
        df_slice = tablepd.loc[tablepd['_sed_wavelen[nm]'] == tablepd.loc[i, '_sed_wavelen[nm]']]
        mean_flux = sum(df_slice['_sed_flux [W/(m^2 Hz)]']) / len(df_slice['_sed_flux [W/(m^2 Hz)]'])
        max_deltaFlux_allowed = rel_error_percentage * mean_flux
        if (tablepd.loc[i, '_sed_flux [W/(m^2 Hz)]'] > (mean_flux + max_deltaFlux_allowed)) or (
                tablepd.loc[i, '_sed_flux [W/(m^2 Hz)]'] < (mean_flux - max_deltaFlux_allowed)):
            index_toBeRemoved.append(i)
    if (tablepd.loc[i + 1, '_sed_wavelen[nm]'] != tablepd.loc[i, '_sed_wavelen[nm]']) and (
            tablepd.loc[i - 1, '_sed_wavelen[nm]'] != tablepd.loc[i, '_sed_wavelen[nm]']):
        m = (tablepd.loc[i + 1, '_sed_flux [W/(m^2 Hz)]'] - tablepd.loc[i - 1, '_sed_flux [W/(m^2 Hz)]']) / (
                tablepd.loc[i + 1, '_sed_wavelen[nm]'] - tablepd.loc[i - 1, '_sed_wavelen[nm]'])
        b = tablepd.loc[i - 1, '_sed_flux [W/(m^2 Hz)]'] - m * tablepd.loc[i - 1, '_sed_wavelen[nm]']
        ideal_flux = m * tablepd.loc[i, '_sed_wavelen[nm]'] + b
        max_deltaFlux_allowed = rel_error_percentage * ideal_flux
        if (tablepd.loc[i, '_sed_flux [W/(m^2 Hz)]'] > (ideal_flux + max_deltaFlux_allowed)) or (
                tablepd.loc[i, '_sed_flux [W/(m^2 Hz)]'] < (ideal_flux - max_deltaFlux_allowed)):
            index_toBeRemoved.append(i)

ax1.errorbar(tablepd['_sed_wavelen[nm]'], tablepd['_sed_flux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']),
             yerr=tablepd['_sed_eflux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']), fmt='.', label='kept')

plt.legend()
plt.show()
plt.close()


# In[4]:


kurucz_data = pd.read_csv('https://wwwuser.oats.inaf.it/castelli/grids/gridm10k2odfnew/fm10t5750g05k2odfnew.dat',
                          skiprows=1, delim_whitespace=True)
print(kurucz_data.columns)

kurucz_data.rename({'VTURB=2': 'model_wavelength[nm]', 'TITLE': '1', '[-1.0]': '2', 'L/H=1.25': '5',
                    'NOVER': 'model_flux [ erg /sec cm ^2 Hz sr]', 'NEW': '6', 'ODF': '7'}, axis=1, inplace=True)
kurucz_data.drop(['1'], axis=1, inplace=True)
print(kurucz_data.columns)

kurucz_data.drop(kurucz_data.tail(1).index, inplace=True)
kurucz_data.drop(kurucz_data.head(30).index, inplace=True)

kurucz_data['model_flux [W/(m^2 Hz sr)]'] = kurucz_data['model_flux [ erg /sec cm ^2 Hz sr]'].astype(float) * 10**-3
kurucz_data.drop(['model_flux [ erg /sec cm ^2 Hz sr]'], axis=1, inplace=True)

# Plot both observed and model data
fig, ax1 = plt.subplots(figsize=(7, 6))
ax1.errorbar(tablepd['_sed_wavelen[nm]'], tablepd['_sed_flux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']),
             yerr=tablepd['_sed_eflux [W/(m^2 Hz)]'] * (tablepd['_sed_freq [Hz]']), fmt='.', color='green', label='Observed Data')
ax1.plot(kurucz_data['model_wavelength[nm]'],
         kurucz_data['model_flux [W/(m^2 Hz sr)]'].astype(float) * (c.value / kurucz_data['model_wavelength[nm]'] * 10**9).astype(float),
         color='blue', marker='', label='Model Data')
ax1.set(xscale='log', yscale='log', xlim=(20, 2e6), ylabel='W / m^2')

ax1.legend(loc='lower right')  # Add legend

plt.show()
plt.close()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Read de-reddening data
de_red_df = pd.read_csv(r'C:\Users\aadis\Desktop\SED\Rv_3_1.dat', skiprows=3, delim_whitespace=True)
de_red_df['Wavelength_deRed [nm]'] = de_red_df['Angstrom'] / 10
de_red_df.drop(['Angstrom'], axis=1, inplace=True)

# Create dataframe with obs, model, de-reddening
tablepd_red_df = pd.merge_asof(tablepd, de_red_df, left_on='_sed_wavelen[nm]', right_on='Wavelength_deRed [nm]', direction='nearest')
kurucz_data.sort_values(by='model_wavelength[nm]', inplace=True)
tablepd_red__model_df = pd.merge_asof(tablepd_red_df, kurucz_data[:-1], left_on='_sed_wavelen[nm]', right_on='model_wavelength[nm]', direction='nearest')
tablepd_red__model_df.drop(columns=['_RAJ2000', '_DEJ2000', '5', '6', '7'], axis=1, inplace=True)

# Interstellar extinction law
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(1 / de_red_df['Wavelength_deRed [nm]'], de_red_df['magn.'], color='blue')
ax.set(xlabel='1/ Wavelength [$nm^{-1}$]', ylabel='$A_{\lambda}/E(B-V)$')
plt.show()


# In[6]:


J_wavelength = 1250
H_wavelength = 1635
K_wavelength = 2190

rescaling_wavelength = H_wavelength
rescale_lambda_index = (tablepd_red__model_df['_sed_wavelen[nm]'] - rescaling_wavelength).abs().argsort()[:1][0]
rescale_lambda = tablepd_red__model_df.loc[rescale_lambda_index, '_sed_wavelen[nm]']
print('The wavelength on which the model will be rescaled: {}, its index is: {}'.format(round(rescale_lambda, 1), rescale_lambda_index))


# In[7]:


import numpy as np

photometry_error = 0.05

chi_square_values = []

def chi_square_compilator(EBV, tablepd_red__model_df, kurucz_data, rescale_lambda_index):
    chi_value = 0
    deRed_fluxes = []
    
    for i in range(len(tablepd_red__model_df)):
        deRed_flux = 10 ** (tablepd_red__model_df.loc[i, 'magn.'] * EBV / 2.5) * tablepd_red__model_df.loc[i, '_sed_flux [W/(m^2 Hz)]']
        deRed_fluxes.append(deRed_flux)
    
    rescale_wavelength_slice = tablepd_red__model_df.loc[tablepd_red__model_df['_sed_wavelen[nm]'] == rescale_lambda]
    rescale_wavelength_list = []
    
    for i in rescale_wavelength_slice.index:
        rescale_wavelength_list.append(deRed_fluxes[i])
    
    avg_rescale_lambda = sum(rescale_wavelength_list) / len(rescale_wavelength_list)
    scaling_factor = avg_rescale_lambda / float(tablepd_red__model_df.loc[rescale_lambda_index, 'model_flux [W/(m^2 Hz sr)]'])
    scaled_model_fluxes = tablepd_red__model_df['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor
    scaled_kuruck_fluxes = kurucz_data['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor
    
    for i in range(len(deRed_fluxes)):
        if tablepd_red__model_df.loc[i, '_sed_wavelen[nm]'] < rescaling_wavelength:
            chi_value += (scaled_model_fluxes[i] - deRed_fluxes[i]) ** 2 / tablepd_red__model_df.loc[i, '_sed_eflux [W/(m^2 Hz)]'] ** 2
    
    return chi_value

EBV_values = np.linspace(0, 2, 1000)

for EBV in EBV_values:
    chi_square_values.append(chi_square_compilator(EBV, tablepd_red__model_df, kurucz_data, rescale_lambda_index))

bestChi = chi_square_values.index(min(chi_square_values))
print(len(chi_square_values), '\nMin chi-square:', round(min(chi_square_values), 2), '\nBest E(B-V) value:', round(EBV_values[bestChi], 4))


# In[23]:


import pickle

with open("ebv_values_results.pkl", "rb") as f:
    ebv_values = pickle.load(f)


# In[22]:


from tqdm import tqdm
import pickle

# Monte-Carlo error simulation
num_trials = 1000
ebv_values = []

for i in tqdm(range(num_trials)):
    tablepd_red__model_df_noisy = tablepd_red__model_df.copy()
    tablepd_red__model_df_noisy['magn.'] += np.random.normal(0, photometry_error, size=len(tablepd_red__model_df))

    chi_square_values_noisy = []

    for EBV in EBV_values:
        chi_square_values_noisy.append(chi_square_compilator(EBV, tablepd_red__model_df_noisy, kurucz_data, rescale_lambda_index))

    bestChi_noisy = chi_square_values_noisy.index(min(chi_square_values_noisy))
    ebv_values.append(EBV_values[bestChi_noisy])

# Save the results using pickle
with open("ebv_values_results.pkl", "wb") as f:
    pickle.dump(ebv_values, f)

# Create a histogram of E(B-V) values
plt.hist(ebv_values, bins=20, alpha=0.6, color='g')
plt.xlabel("E(B-V)")
plt.ylabel("Frequency")
plt.show()


# In[9]:


subset_size = 10  # Number of trials to plot, adjust as needed
selected_indices = np.random.choice(num_trials, subset_size, replace=False)

plt.figure(figsize=(10, 6))
for index in selected_indices:
    plt.plot(EBV_values, all_chi_square_values[index], label=f'Trial {index}')

plt.xlabel('E(B-V)')
plt.ylabel('Chi-square value')
plt.yscale('log') 
plt.legend()
plt.show()


# In[8]:


import pickle

# Load the E(B-V) values from the pickle file
with open("ebv_values_results.pkl", "rb") as f:
    EBV_values = pickle.load(f)

# Compute chi-square for the loaded E(B-V) values
chi_square_values = []
for EBV in EBV_values:
    chi_square_values.append(chi_square_compilator(EBV, tablepd_red__model_df, kurucz_data, rescale_lambda_index))

# Find the color index corresponding to the lowest chi-square value
min_chi_square_index = np.argmin(chi_square_values)
best_EBV = EBV_values[min_chi_square_index]

# Plot the chi-square values and highlight the best E(B-V)
plt.figure(figsize=(10, 6))
plt.plot(EBV_values, chi_square_values, 'b-', label='Chi-square values')
plt.plot(best_EBV, min(chi_square_values), 'ro', label=f'Best E(B-V): {best_EBV:.4f}')
plt.xlabel('E(B-V)')
plt.ylabel('Chi-square value')
plt.yscale('log') 
plt.legend()

plt.show()


# In[25]:


mean_EBV = np.mean(EBV_values)
std_EBV = np.std(EBV_values)
print("Mean E(B-V) value:", round(mean_EBV, 4))
print("Standard deviation of E(B-V) values:", round(std_EBV, 4))


# In[26]:


# Calculate the mean and standard deviation of the EBV values
mean_EBV = np.mean(EBV_values)
std_EBV = np.std(EBV_values)

# Plot the histogram of EBV values
plt.figure(figsize=(10, 6))
plt.hist(EBV_values, bins=20, color='blue', alpha=0.7)
plt.axvline(x=mean_EBV, color='red', linestyle='--')
plt.axvline(x=mean_EBV + std_EBV, color='green', linestyle='--')
plt.axvline(x=mean_EBV - std_EBV, color='green', linestyle='--')
plt.xlabel('E(B-V)')
plt.ylabel('Frequency')
plt.legend()

plt.show()


# In[27]:


deRed_fluxes = []
for i in range(len(tablepd_red__model_df)):
    deRed_flux = 10 ** (tablepd_red__model_df.loc[i, 'magn.'] * best_EBV / 2.5) * tablepd_red__model_df.loc[i, '_sed_flux [W/(m^2 Hz)]']
    deRed_fluxes.append(deRed_flux)

rescale_wavelength_slice = tablepd_red__model_df.loc[tablepd_red__model_df['_sed_wavelen[nm]'] == rescale_lambda]
rescale_wavelength_list = []
for i in rescale_wavelength_slice.index:
    rescale_wavelength_list.append(deRed_fluxes[i])

avg_rescale_lambda = sum(rescale_wavelength_list) / len(rescale_wavelength_list)
scaling_factor = avg_rescale_lambda / float(tablepd_red__model_df.loc[rescale_lambda_index, 'model_flux [W/(m^2 Hz sr)]'])
scaled_model_fluxes = tablepd_red__model_df['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor
scaled_kuruck_fluxes = kurucz_data['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor

tablepd_red__model_df['scaled_model_fluxes'] = scaled_kuruck_fluxes
tablepd_red__model_df['deRed_data_fluxes'] = deRed_fluxes

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(kurucz_data['model_wavelength[nm]'], scaled_kuruck_fluxes * (c * 10 ** 9 / kurucz_data['model_wavelength[nm]']), color='green', marker='', label='SED model')
ax.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], tablepd_red__model_df['_sed_flux [W/(m^2 Hz)]'].astype(float) * tablepd_red__model_df['_sed_freq [Hz]'], marker='.', color='red', label='measured')
ax.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], deRed_fluxes * tablepd_red__model_df['_sed_freq [Hz]'], marker='.', color='blue', label='dereddened')
ax.set(xscale='log', yscale='log', ylim=(3e-15, 1e-13), xlim=(200, 1e5), title='SED (model rescaled at {} nm)'.format(round(rescale_lambda, 2)),
       xlabel='Wavelength [nm]', ylabel='Flux ($F_{\\nu}\ \\nu$) [$W m^{-2}$]')

ax.axvline(x=rescale_lambda)
ax.text(rescale_lambda, 7.e-14, '{} nm'.format(round(rescale_lambda, 1)))
ax.legend()

plt.show()
plt.close()


# In[14]:


EBV_finalValue = mean_EBV
deRed_fluxes = []
for i in range(len(tablepd_red__model_df)):
    deRed_flux = 10 ** (tablepd_red__model_df.loc[i, 'magn.'] * EBV_finalValue / 2.5) * tablepd_red__model_df.loc[i, '_sed_flux [W/(m^2 Hz)]']
    deRed_fluxes.append(deRed_flux)

rescale_wavelength_slice = tablepd_red__model_df.loc[tablepd_red__model_df['_sed_wavelen[nm]'] == rescale_lambda]
rescale_wavelength_list = []
for i in rescale_wavelength_slice.index:
    rescale_wavelength_list.append(deRed_fluxes[i])

avg_rescale_lambda = sum(rescale_wavelength_list) / len(rescale_wavelength_list)
tablepd_red__model_df['deRed_data_fluxes'] = deRed_fluxes
scaling_factor = avg_rescale_lambda / float(tablepd_red__model_df.loc[rescale_lambda_index, 'model_flux [W/(m^2 Hz sr)]'])
tablepd_red__model_df['scaled_model_fluxes'] = tablepd_red__model_df['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor
scaled_kuruck_fluxes = kurucz_data['model_flux [W/(m^2 Hz sr)]'].astype(float) * scaling_factor

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(kurucz_data['model_wavelength[nm]'], scaled_kuruck_fluxes * (c * 10 ** 9 / kurucz_data['model_wavelength[nm]']), color='yellow', marker='', label='SED model')
ax.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], tablepd_red__model_df['_sed_flux [W/(m^2 Hz)]'].astype(float) * tablepd_red__model_df['_sed_freq [Hz]'], marker='.', color='red', label='measured')
ax.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], deRed_fluxes * tablepd_red__model_df['_sed_freq [Hz]'], marker='.', color='blue', label='dereddened')
ax.set(xscale='log', yscale='log', ylim=(3e-15, 1e-13), xlim=(200, 1e5), title='SED (model rescaled at {} nm)'.format(round(rescale_lambda, 2)),
       xlabel='Wavelength [nm]', ylabel='Flux ($F_{\\nu}\ \\nu$) [$W m^{-2}$]')

ax.axvline(x=rescale_lambda)
ax.text(rescale_lambda, 7.e-14, '{} nm'.format(round(rescale_lambda, 1)))
ax.legend()

plt.show()
plt.close()


# In[33]:


tot_flux = abs(np.trapz(scaled_kuruck_fluxes, x=(c.value / (kurucz_data['model_wavelength[nm]'] * 10**-9)))) * 10**3  # converted to erg / (s cm^2)
print("Total Flux:", tot_flux)

distance = 61.00e3  # in pc
solar_lum = 3.846e33  # in erg/s
solar_distance = 4.77e-6  # in parsec
solar_flux = 1.4e6  # in erg/s
stellar_Temperature = 5750  # in K
sigmaSB = 5.670374e-5  # erg / (cm^2 s K^4)
solar_radius = 6.963e8  # in m

luminosity = (solar_lum * distance**2 * tot_flux) / (solar_distance**2 * solar_flux)
print("Luminosity (erg/sec):", luminosity)

stellar_radius = math.sqrt(luminosity / (math.pi * 4 * sigmaSB * stellar_Temperature**4)) * 10**-2  # converted to meters
print("Luminosity (4*pi*R^2*F):", 4 * np.pi * distance**2 * tot_flux)

print("The luminosity of the star is: {:.3e} erg/sec".format(luminosity))
print("The luminosity of the star is: {:.3} solar luminosity".format(luminosity / solar_lum))
print("The radius of the star is: {:.3e} m, or: {:.3} solar radii".format(stellar_radius, stellar_radius / solar_radius))


# In[34]:


IR_emission_values = tablepd_red__model_df['deRed_data_fluxes'] - tablepd_red__model_df['scaled_model_fluxes']

# Plot the flux points and fitted curve
fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], tablepd_red__model_df['_sed_flux [W/(m^2 Hz)]'].astype(float) * tablepd_red__model_df['_sed_freq [Hz]'],
            marker='.', color='red', label='Observed Flux')
ax1.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], deRed_fluxes * tablepd_red__model_df['_sed_freq [Hz]'],
            marker='.', color='blue', label='Dereddened Flux')
ax1.plot(kurucz_data['model_wavelength[nm]'], scaled_kuruck_fluxes * (c * 10**9 / kurucz_data['model_wavelength[nm]']), color='yellow', marker='.', label='Model Flux')
ax1.scatter(tablepd_red__model_df['_sed_wavelen[nm]'], IR_emission_values * tablepd_red__model_df['_sed_freq [Hz]'], marker='.', color='purple', label='IR Emission')
ax1.set(xscale='log', yscale='log', ylim=(1e-15, 1e-13), xlim=(200, 1e5))
ax1.legend()
plt.show()
plt.close()

# Define the Planck function
def plancks_law(wavelength, T, const):
    b = 1.38e-23
    h = 6.626e-34
    freq = c.value / (wavelength * 1e-9)
    I = h * const / c.value**2 * freq**3 / (np.exp((h * freq) / (b * T)) - 1)  # fitting a Planck function
    return I

# Fit the Planck function to estimate dust temperature
popt, pcov = curve_fit(plancks_law, tablepd_red__model_df['_sed_wavelen[nm]'], IR_emission_values * tablepd_red__model_df['_sed_freq [Hz]'], p0=(277, 1e-4))
perr = np.sqrt(np.diag(pcov))

print('The dust has a temperature of approx: {} +- {} K'.format(round(popt[0], 0), round(perr[0], 0)))
print('The constant has a value of {} +- {}'.format(round(popt[1], 6), round(perr[1], 6)))


# In[40]:


# Distance of the disk from the star
dust_temperature = popt[0]
star_dust_distance = stellar_radius / 2 * (stellar_Temperature / dust_temperature) ** 2
print('The distance of the bulk of the dust from the star is approx: {:.4e} m'.format(star_dust_distance))
print('The distance of the bulk of the dust from the star is approx: {:.4e} solar radii'.format(star_dust_distance / solar_radius))

IR_xNew = np.logspace(3, 5, 1000)
IR_yNew = plancks_law(IR_xNew, *popt)

max_index = list(IR_yNew).index(max(list(IR_yNew)))
lambda_max = list(IR_xNew)[max_index]
const = 2.9e6  # in nm
temp = const / lambda_max
print('The temperature is: {:.1f}'.format(temp))

# Filter the data points with wavelengths larger than 2200 nm
filtered_wavelengths = tablepd_red__model_df['_sed_wavelen[nm]'][tablepd_red__model_df['_sed_wavelen[nm]'] > 2200]
filtered_fluxes = IR_emission_values[tablepd_red__model_df['_sed_wavelen[nm]'] > 2200] * tablepd_red__model_df['_sed_freq [Hz]'][tablepd_red__model_df['_sed_wavelen[nm]'] > 2200]

# Plot the filtered flux points and fitted curve
fig, ax1 = plt.subplots(figsize=(8, 8))
ax1.scatter(filtered_wavelengths, filtered_fluxes, marker='.', color='purple', label='dust flux')
ax1.plot(IR_xNew, IR_yNew, label='Planck curve', color='blue')
ax1.set(xscale='log', yscale='log', xlabel='Wavelength [nm]', ylabel='Flux ($F_{\\nu}\\nu$) [$W m^{-2}$]')
ax1.axvline(x=lambda_max, linestyle='--', color='black', linewidth=1)
ax1.text(lambda_max, 1e-16, '$\\lambda_{max}$')
ax1.legend()

plt.show()


# In[36]:


max_index = np.argmax(IR_yNew)
lambda_max = IR_xNew[max_index]
const = 2.9e6  # in nm
temp = const / lambda_max
print('The temperature is: {:.1f} K'.format(temp))

# Luminosity of the dust
tot_dust_flux = abs(np.trapz(IR_yNew / (c.value / (IR_xNew * 10**-9)), x=(c.value / (IR_xNew * 10**-9)))) * 10**3  # converted to erg/s
Luminosity_dust = solar_lum * distance**2 * tot_dust_flux / (solar_distance**2 * solar_flux)
print('The luminosity of the dust is: {:.3e} erg/sec'.format(Luminosity_dust))
print('The luminosity of the dust is: {:.3} solar luminosity'.format(Luminosity_dust / solar_lum))
print('The luminosity of the dust is: {:.3e}'.format(Luminosity_dust))


# In[ ]:




