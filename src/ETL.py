import pandas as pd
import numpy as np
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


def create_palier(df: pd.DataFrame, cutoff_date, npal: int = 4):

    def _sub_pal_num(data_):
        col_pal = [col for col in df if 'PAL' in col]
        data_ = data_.rename(columns=dict(zip(col_pal,
                                              [re.sub('PAL[0-9]', 'PAL', col) for col in col_pal])))
        return data_

    PAL = pd.concat([_sub_pal_num(df[['NDOSS'] + [col for col in df if col.endswith('PAL' + str(i))]]).assign(PAL=i)
                     for i in range(1, npal+1)])

    PAL = PAL.rename(columns={'FUNIIMS_DT_PREM_ECH_PAL': 'PAL_DStart',
                              'FUNIIMS_MT_MENS_PAL': 'Mensualite'})

    PAL = PAL.dropna(subset='PAL_DStart')

    return PAL.sort_values(by=['PAL_DStart', 'NDOSS'])


def create_model(df: pd.DataFrame, pal: pd.DataFrame, col_dates):

    model = df.join(pd.DataFrame(col_dates).rename(columns={0: 'DateProj'}), how='cross')\
              .sort_values(by=['DateProj', 'NDOSS'])

    model = pd.merge_asof(model, pal[['NDOSS', 'PAL_DStart', 'Mensualite']],
                         left_on='DateProj', right_on='PAL_DStart', by='NDOSS').drop(columns=['PAL_DStart'])

    return model


def output_model(model: pd.DataFrame, col_index, col_dates):

    model[col_index] = model[col_index].replace(pd.NaT, pd.to_datetime('today')).fillna('NAN')
    out_ = pd.pivot_table(model,
                          values='Mensualite',
                          index=col_index,
                          columns=['DateProj'],
                          aggfunc=np.nansum,
                          fill_value=0,
                          observed=True,
                          ).reset_index().replace("NAN", np.nan)

    out_['Mens_total']      = out_[col_dates].sum(axis=1)
    out_['Sum Mens - CRD']  = out_['Mens_total'] - out_['CTX_MCRD']

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

    paliers     = create_palier(dt, config.dCutoff)

    model       = create_model(dt, paliers, config.col_dates)

    output      = output_model(model, list(dt.columns), config.col_dates)
