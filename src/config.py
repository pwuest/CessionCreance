import os
from types import SimpleNamespace

import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

SRC_DIR     = os.path.dirname(__file__)
ROOT_DIR    = os.path.join(SRC_DIR, os.pardir)
DATA_DIR    = os.path.join(ROOT_DIR, 'data')

dCutoff     = '2021-12-31'
dCutoff     = pd.to_datetime(dCutoff)
nHorizon    = 300

col_dates = [dCutoff + relativedelta(months=i) + MonthEnd(0) for i in range(1, nHorizon+1)]
# col_dates = [dat.date() for dat in col_dates]

datatape_path = SimpleNamespace(
    io=os.path.join(DATA_DIR, 'Presentation_20220303_fichier cession sdt 2022 via CRE CRI.xlsx'),
    sheet_name='CESSION_SDT_2022'
)

path_cache      = os.path.join(DATA_DIR, 'datatape.pkg')
path_result     = os.path.join(DATA_DIR, 'model.xlsx')

