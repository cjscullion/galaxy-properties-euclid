import numpy as np
import pandas as pd
import math

np.random.seed(seed = 45)

observable_df = pd.read_csv('observable_full.csv')

limiting_mags = {
    'LSST_u': 24.2,
    'LSST_g': 24.5,
    'LSST_r': 23.9,
    'LSST_i': 23.6,
    'LSST_z': 23.4,
    'VIS': 24.5,
    'NISP_Y': 23.0,
    'NISP_J': 23.0,
    'NISP_H': 23.0
}

upper_mag = (observable_df['VIS'] < limiting_mags['VIS'])
observable_df = observable_df[upper_mag]

lower_mag = (observable_df['VIS'] > 15)
observable_df = observable_df[lower_mag]

def AB_to_flux(AB):
    f=np.ones(len(AB))
    f = pow(10.0, -0.4*(np.asarray(AB) - 23.9))
    return f

def flux_to_AB(flux):
    m=-2.5*np.log10(flux)+23.9
    np.nan_to_num(m,copy=False,nan=99.0,posinf=None,neginf=None)
    return m

limiting_fluxes = limiting_mags.copy()

for i in limiting_fluxes.keys():
    limiting_fluxes[i] = 10**(-0.4*(limiting_mags[i] - 23.9))

lim_flux_err = limiting_fluxes.copy()

SNR = 10

for i in lim_flux_err.keys():
    lim_flux_err[i] = lim_flux_err[i]/SNR

lim_mag_err = lim_flux_err.copy()

for i in lim_flux_err.keys():
    lim_mag_err[i] = (2.5/math.log(10.))*np.abs(lim_flux_err[i]/limiting_fluxes[i])

phot_errors_df = observable_df.copy()

def addFluxColumns(table):
    row_number = table["VIS"].size
    for name in lim_flux_err.keys():
        col_data = AB_to_flux(table[name])
        col_name = 'FLUX_'+name
        table[col_name]=col_data
    return table

def addErrorColumns(table):
    row_number = table["VIS"].size
    for name in lim_flux_err.keys():
        col_data = np.ones(row_number)*lim_flux_err.get(name)
        col_name = 'FLUX_'+name+'_ERR'
        table[col_name]=col_data
    return table

def randomizeFlux(table):
    row_number = table['VIS'].size
    for name in lim_flux_err.keys():
        s = np.random.normal(0,table['FLUX_'+name+'_ERR'],row_number)
        table['FLUX_'+name+'_OBS']=table['FLUX_'+name]+s
    return table

def addObsMagnitudes(table):
    for name in lim_flux_err.keys():
        flux_obs=table['FLUX_'+name+'_OBS']
        mag_obs = flux_to_AB(flux_obs)
        dmag = (2.5/math.log(10.))*np.abs(table['FLUX_'+name+'_ERR']/table['FLUX_'+name])
        table[name+'_OBS'] = mag_obs
        table['delta_'+ name] = dmag
        table.loc[table[name]>90., ('delta_'+ name)] = 99.
    return table

# very weird here
table_with_flux = addFluxColumns(phot_errors_df)
table_with_error = addErrorColumns(phot_errors_df)
del phot_errors_df
randomized_table = randomizeFlux(table_with_error)
del table_with_error
phot_errors_df = addObsMagnitudes(randomized_table)

sample_df = phot_errors_df.sample(6000, random_state=45).copy()

sample_df.to_csv('euc_obs_n6000.csv')

full_filts = sample_df.drop(columns=['For_key_ID', 'LSST_u', 'LSST_g', 'LSST_r', 'LSST_i', 'LSST_z', 'VIS',
       'NISP_Y', 'NISP_J', 'NISP_H', 'z', 't/Gyr', 'M*', 'M_initial', 'Z',
       'SFH', 'tau/Gyr', 'Av_law', 'Av', 't_l', 'physical', 'FLUX_LSST_u',
       'FLUX_LSST_g', 'FLUX_LSST_r', 'FLUX_LSST_i', 'FLUX_LSST_z', 'FLUX_VIS',
       'FLUX_NISP_Y', 'FLUX_NISP_J', 'FLUX_NISP_H', 'FLUX_LSST_u_ERR',
       'FLUX_LSST_g_ERR', 'FLUX_LSST_r_ERR', 'FLUX_LSST_i_ERR',
       'FLUX_LSST_z_ERR', 'FLUX_VIS_ERR', 'FLUX_NISP_Y_ERR', 'FLUX_NISP_J_ERR',
       'FLUX_NISP_H_ERR', 'FLUX_LSST_u_OBS', 'FLUX_LSST_g_OBS',
       'FLUX_LSST_r_OBS', 'FLUX_LSST_i_OBS', 'FLUX_LSST_z_OBS', 'FLUX_VIS_OBS',
       'FLUX_NISP_Y_OBS', 'FLUX_NISP_J_OBS', 'FLUX_NISP_H_OBS'])

full_filts['z'] = sample_df['z']

full_filts.insert(loc=0, column='ID', value=np.arange(1, len(full_filts)+1))

np.savetxt(r'all_n6000.cat', full_filts.values, fmt=' '.join(['%i'] + ['%1.4f']*19))

no_u_cat = full_filts.drop(columns=['LSST_u_OBS', 'delta_LSST_u'])

np.savetxt(r'no_u_n6000.cat', no_u_cat.values, fmt=' '.join(['%i'] + ['%1.4f']*17))

gnd_cat = full_filts.drop(columns=['VIS_OBS',
       'NISP_Y_OBS', 'NISP_J_OBS', 'NISP_H_OBS','delta_VIS', 'delta_NISP_Y', 'delta_NISP_J', 'delta_NISP_H'])

np.savetxt(r'gnd_n6000.cat', gnd_cat.values, fmt=' '.join(['%i'] + ['%1.4f']*11))

euc_cat = full_filts.drop(columns=['LSST_u_OBS', 'LSST_g_OBS', 'LSST_r_OBS', 'LSST_i_OBS', 'LSST_z_OBS', 'delta_LSST_u',
       'delta_LSST_g', 'delta_LSST_r', 'delta_LSST_i', 'delta_LSST_z'])

np.savetxt(r'euc_n6000.cat', euc_cat.values, fmt=' '.join(['%i'] + ['%1.4f']*9))
