from datetime import datetime

import numpy as np
import pandas as pd

from ficc.utils.auxiliary_variables import IS_SAME_DAY
from ficc.utils.adding_flags import add_same_day_flag


df = pd.DataFrame({'trade_date': [datetime(2022, 7, 14),  
                                  datetime(2022, 7, 14)], 
                   'price': [110.09, 
                             109.09], 
                   'par_traded': [100, 
                                  100], 
                   'trade_type': ['S', 
                                  'P'], 
                   'cusip': [1, 1]})

df = add_same_day_flag(df, use_parallel_apply=False)
assert np.array_equal(df[IS_SAME_DAY].values, [True, True])