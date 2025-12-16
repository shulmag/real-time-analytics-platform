import numpy as np
import pandas as pd
from datetime import datetime 
from pytz import timezone

from auxiliary_functions import resolve_date, date_hour_format, load_yield_curve_params, predict_ytw


eastern = timezone('US/Eastern')

##### Base YieldCurve Class
class YieldCurve():
    def __init__(self, **kwargs):
        self._curve_type = None
        self._yield_curves = None
        self._initialized = False
        self._yield_curve_params = None
        
        if kwargs.get('start_date'):
            self.start_date = resolve_date(kwargs['start_date'])
            print(f'Setting start date to {self.start_date}')
        else:    
            self.start_date = eastern.localize(datetime.strptime('2021-07-27 00:00', '%Y-%m-%d %H:%M'))
            print('No argument start_date given, defaulting to 2021-07-27 00:00 ET')
               
        if kwargs.get('end_date'): 
            self.end_date = resolve_date(kwargs['end_date'])     
            print(f'Setting end date to {self.end_date}')
        else: 
            self.end_date = datetime.now(eastern)
            print(f'No argument end_date given, defaulting to the time now, {self.end_date.strftime(date_hour_format)} ET')            
            
        if kwargs.get('setting'):
            self.setting = kwargs.get('setting')       
        else:
            self.setting = 'DAILY'        

        if 't' in kwargs: 
            t = kwargs.get('t')
            if isinstance(t, float):
                self.t = np.array(t).reshape(-1,1)
            
            elif isinstance(t, np.ndarray):
                self.t = np.array(t).reshape(-1,1)
            
            else:
                try: 
                    self.t = np.array(t).astype(float)
                except TypeError:
                    print('Invalid maturities given. Defaulting to 0.1 to 30')
                    
        else: self.t = np.round(np.arange(0.1, 30.1, 0.1),2)        
        
    def get_curve_type(self):
        return self._curve_type
    
    def update_data(self, **kwargs):            
        if not self._yield_curve_params.empty:            
            if kwargs.get('start_date'): start_date = resolve_date(kwargs['start_date'])
            else: start_date = self.start_date
        
            if kwargs.get('end_date'): end_date = resolve_date(kwargs['end_date'])     
            else: end_date = datetime.now(eastern)
            
            print(f'Update yield curve data from ({self.start_date.strftime("%Y-%m-%d")} to {self.end_date.strftime("%Y-%m-%d")}) to ({start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}), Y/N')
            confirmation = input()
            
            if (confirmation == 'Y') or (confirmation =='y'):  
                self.start_date = start_date
                self.end_date = end_date 
                self.initialize_data()
            else:
                print('Update cancelled')
        
        else: self.initialize_data()
    
    def initialize_data(self):
        pass
    
    def predict_yields(self, **kwargs):
        if not self._initialized:
            print('Nothing to display, yield curve not initialized.')
            return None
        
        if 't' in kwargs: 
            t = kwargs.get('t')
            if isinstance(t, float):
                t = np.array(t).reshape(-1,1)
            
            elif isinstance(t, np.ndarray):
                t = np.array(t).reshape(-1,1).astype(float)
            
            else:
                try: 
                    t = np.array(t).astype(float)
                except TypeError:
                    print('Invalid maturities given. Defaulting to 0.1 to 30') 
        
        else: t = self.t
    
        print(f'Estimating yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        temp = self._yield_curve_params.apply(lambda x: predict_ytw(t, 
                                                                x.const, 
                                                                x.exponential, 
                                                                x.laguerre, 
                                                                x.exponential_mean, 
                                                                x.exponential_std,
                                                                x.laguerre_mean,
                                                                x.laguerre_std,
                                                                 x.L),
                                                          axis=1)
        
        temp = pd.DataFrame(zip(*temp)).T
        temp.index = self._yield_curve_params.index.values
        temp.columns = t.flatten()
        return temp
   
    def get_yield_curves(self):
        if not self._initialized:
            print('Nothing to display, yield curve not initialized. Call self.initialize_data()')
            return None
        return self._yield_curves

    def get_yield_curve_params(self):
        if not self._initialized:
            print('Nothing to display, yield curve not initialized. Call self.initialize_data()')
            return None
        return self._yield_curve_params
    
    def visualize():
        pass
    
    def fit(self, **kwargs):        
        if not self._initialized:
            print('Nothing to display, yield curve not initialized.')
            return None
        
        print(f'Estimating and saving yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        temp = self._yield_curve_params.apply(lambda x: predict_ytw(self.t, 
                                                                x.const, 
                                                                x.exponential, 
                                                                x.laguerre, 
                                                                x.exponential_mean, 
                                                                x.exponential_std,
                                                                x.laguerre_mean,
                                                                x.laguerre_std,
                                                                x.L),
                                                          axis=1)
        
        temp = pd.DataFrame(zip(*temp)).T
        temp.index = self._yield_curve_params.index.values
        temp.columns = self.t.flatten()
        self._yield_curves = temp
        self._initialized = True


class DailyYieldCurveBase(YieldCurve): 
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self._curve_type = 'DAILY'
        
    def initialize_data(self, **kwargs):
        print(f'Loading daily yield curve values from BigQuery')
        self._yield_curve_params = load_yield_curve_params(daily=True, 
                                                          date_start= self.start_date.strftime('%Y-%m-%d'), 
                                                          date_end = self.end_date.strftime('%Y-%m-%d'), 
                                                          use_redis=False)     
        self._initialized = True
        print(f'Daily yield curve values loaded, data spans {self._yield_curve_params.index.min()} to {self._yield_curve_params.index.max()}')
        print(f'Estimating yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        super().fit(**kwargs)
        
    # def visualize(self, use_plotly = False, print_summary=False, start = None, end = None,  xlim = 30, ylim_lower = 270, ylim_upper = 350):
    #     if not self._initialized:
    #         print('Nothing to plot, yield curve not initialized.')
        
    #     return plot_curves(self._yield_curves, 
    #                 daily = True, 
    #                 use_plotly = use_plotly, 
    #                 print_summary=print_summary, 
    #                 start = start, 
    #                 end = end, 
    #                 xlim = 30, 
    #                 ylim_lower = 270, 
    #                 ylim_upper = 350,
    #                        return_figure=return_figure)


class RealTimeYieldCurve(YieldCurve):
    def __init__(self, **kwargs):
        super().__init__( **kwargs) 
        self._curve_type = 'REALTIME'
        
    def initialize_data(self, **kwargs):
        if kwargs.get('use_redis'): use_redis = True
        else: use_redis = False
        
        print(f'Loading real time yield curve values from {((1-use_redis)*"BigQuery" + use_redis *"Redis")}')
        self._yield_curve_params = load_yield_curve_params(daily=False, 
                                                           date_start= self.start_date.strftime('%Y-%m-%d'), 
                                                           date_end = self.end_date.strftime('%Y-%m-%d'), 
                                                           use_redis=use_redis)
        self._initialized = True
        print(f'Real time yield curve values loaded, data spans {self._yield_curve_params.index.min()} to {self._yield_curve_params.index.max()}')
        print(f'Estimating yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        super().fit(**kwargs)
        
        
    # def visualize(self, date, use_plotly = False, print_summary=False, xlim = 30, ylim_lower = 270, ylim_upper = 350, return_figure=False):
    #     if not self.initialized:
    #         print('Nothing to plot, yield curve not initialized.')

    #     return plot_curves(self._yield_curves, 
    #                         daily = False,
    #                         date = date,
    #                         use_plotly = use_plotly, 
    #                         print_summary=print_summary, 
    #                         start = None, 
    #                         end = None, 
    #                         xlim = xlim, 
    #                         ylim_lower = ylim_lower, 
    #                         ylim_upper = ylim_upper,
    #                       return_figure=return_figure)
