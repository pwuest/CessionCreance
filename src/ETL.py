import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

import config
import re
import pickle


def extract_datatape(datatape_path):
    usecols     = ['NDOSS', 'Libelle_Strate', 'ENCAISSEMENT_MENSUEL', 'PRODUIT', 'CTX_MPLAN', 'CTX_MEFFMT_SRD',
                   'CTX_MCRD', 'CTX_MSOLD', 'CTX_MMT_ENCAISS_AN', 'CTX_MMT_ENCAISS_ORIG', 'CTX_DDERN_ENCAISS']
    df          = pd.read_excel(**vars(datatape_path),
                                usecols=lambda x: x if x.startswith('FUNIIMS') or x in usecols else None
                                )

    df          = df.loc[~df['NDOSS'].isna(), :]
    df['NDOSS'] = df['NDOSS'].astype(str).str.replace('.0', '', regex=False)

    col_dates = [col for col in df.columns if 'DT_' in col] + ['CTX_DDERN_ENCAISS']
    df[col_dates] = df[col_dates].apply(pd.to_datetime, errors='coerce', dayfirst=True)

    return df


def create_palier(df: pd.DataFrame, npal: int = 4):

    def _sub_pal_num(data_):
        col_pal = [col for col in df if 'PAL' in col]
        data_ = data_.rename(columns=dict(zip(col_pal,
                                              [re.sub('PAL[0-9]', 'PAL', col) for col in col_pal])))
        return data_

    PAL = pd.concat([_sub_pal_num(df[['NDOSS'] + [col for col in df if col.endswith('PAL' + str(i))]]).assign(PAL=i)
                     for i in range(1, npal+1)])

    PAL = PAL.rename(columns={'FUNIIMS_DT_PREM_ECH_PAL': 'PAL_DStart',
                              'FUNIIMS_MT_MENS_PAL': 'Mensualite',
                              'FUNIIMS_NBR_MENS_PAL': 'NbrMensualites'})

    PAL = PAL.dropna(subset='PAL_DStart').sort_values(by=['NDOSS', 'PAL'], ascending=False)
    PAL['NbrMensualites'] = PAL['NbrMensualites'].fillna(0)

    # Fin de plan
    FdP = PAL.copy().groupby('NDOSS').first().reset_index()
    FdP['DateFinPlan'] = FdP.apply(lambda x: x['PAL_DStart'] + relativedelta(months=max(int(x['NbrMensualites'])-1, 0))
                                             + MonthEnd(0), axis=1)

    PAL                = pd.merge(PAL, FdP[['NDOSS', 'DateFinPlan']], on='NDOSS')

    return PAL.sort_values(by=['PAL_DStart', 'NDOSS'])


def create_model(df: pd.DataFrame, pal: pd.DataFrame, col_dates):

    model = df.join(pd.DataFrame(col_dates).rename(columns={0: 'DateProj'}), how='cross')\
              .sort_values(by=['DateProj', 'NDOSS'])

    model = pd.merge_asof(model, pal[['NDOSS', 'PAL_DStart', 'Mensualite', 'DateFinPlan']],
                         left_on='DateProj', right_on='PAL_DStart', by='NDOSS').drop(columns=['PAL_DStart'])

    model.loc[model['DateProj'] > model['DateFinPlan'], 'Mensualite'] = 0
    model['DateFinPlan'] = model.groupby('NDOSS')[['DateFinPlan']].fillna(method='bfill')
    model['Mensualite']  = model['Mensualite'].fillna(0)

    return model


def output_model(model: pd.DataFrame, col_index, col_dates, cutoff_date):

    model[col_index] = model[col_index].replace(pd.NaT, pd.to_datetime('1900-01-01')).fillna('NAN')
    out_ = pd.pivot_table(model,
                          values='Mensualite',
                          index=col_index,
                          columns=['DateProj'],
                          aggfunc=np.nansum,
                          fill_value=0,
                          observed=True,
                          ).reset_index().replace("NAN", np.nan).replace(pd.to_datetime('1900-01-01'), None)

    out_['Mens_total']            = out_[col_dates].sum(axis=1)
    out_['Check Sum Mens - CRD']  = out_['Mens_total'] - out_['CTX_MCRD']
    out_['DateDebutPlan']         = out_['FUNIIMS_DT_PREM_ECH_PAL1']
    out_['Seasoning']             = (cutoff_date - out_['DateDebutPlan'])/np.timedelta64(1, 'M')
    out_['Seasoning']             = out_['Seasoning'].fillna(0).apply(np.floor).astype(int)

    # reshuffle and rename columns
    out_ = out_[[col for col in out_.columns if col not in col_dates] + col_dates]
    out_ = out_.rename(columns=dict(zip(col_dates, [dat.date() for dat in col_dates])))

    col_dat = [col for col in out_.columns if 'DT_' in str(col)] + ['CTX_DDERN_ENCAISS', 'DateFinPlan', 'DateDebutPlan']
    out_[col_dat] = out_[col_dat].apply(lambda x: pd.to_datetime(x, errors='coerce', dayfirst=True).dt.date)

    return out_



if __name__ == "__main__":

    # dt = extract_datatape(config.datatape_path)
    #
    # # Save / Cache data for faster reload if need be
    # with open(config.path_cache, "wb") as file:
    #     pickle.dump(dt, file)
    #     file.close()

    with open(config.path_cache, "rb") as f:
        dt = pickle.load(f)
        f.close()

    paliers         = create_palier(dt)

    modelCF         = create_model(dt, paliers, config.col_dates)

    output          = output_model(modelCF,
                                   list(dt.columns) + ['DateFinPlan'],
                                   config.col_dates, config.dCutoff)

    print("Done")

    output.to_excel(config.path_result, index=False, sheet_name="Model")
