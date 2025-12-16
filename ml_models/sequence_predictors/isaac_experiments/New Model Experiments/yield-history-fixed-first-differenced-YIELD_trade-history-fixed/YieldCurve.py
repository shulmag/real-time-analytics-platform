from auxiliary_functions import *

##### Base YieldCurve Class
class YieldCurve():
    def __init__(self, **kwargs):
        
        self.curve_type = None
        self.index_data = None
        self.maturity_data = None
        self.etf_data = None
        
        self.yield_curves = None
        self.initialized = False
        self.yield_curve_params = None
        
        
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
        
    def update_data(self, **kwargs):            
        if not self.yield_curve_params.empty:            
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
        
        if not self.initialized:
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
        temp = self.yield_curve_params.apply(lambda x: predict_ytw(t, 
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
        temp.index = self.yield_curve_params.index.values
        temp.columns = t.flatten()
        return temp
   
    def get_yield_curve_params(self):
        return self.yield_curve_params
    
    def visualize():
        pass
    
    def fit(self, **kwargs):        
        if not self.initialized:
            print('Nothing to display, yield curve not initialized.')
            return None
        
        print(f'Estimating and saving yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        temp = self.yield_curve_params.apply(lambda x: predict_ytw(self.t, 
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
        temp.index = self.yield_curve_params.index.values
        temp.columns = self.t.flatten()
        self.yield_curves = temp
        self.initialized = True
        
        
    def load_index_data(self):
        self.index_data = get_index_data(self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))
    
    def load_maturity_data(self):
        self.maturity_data = get_index_data(self.start_date.strftime('%Y-%m-%d'), self.end_date.strftime('%Y-%m-%d'))

class DailyYieldCurveBase(YieldCurve): 
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.curve_type = 'DAILY'
        
    def initialize_data(self, **kwargs):
        print(f'Loading daily yield curve values from BigQuery')
        self.yield_curve_params = load_yield_curve_params(daily=True, 
                                                          date_start= self.start_date.strftime('%Y-%m-%d'), 
                                                          date_end = self.end_date.strftime('%Y-%m-%d'), 
                                                          use_redis=False)     
        self.initialized = True
        print(f'Daily yield curve values loaded, data spans {self.yield_curve_params.index.min()} to {self.yield_curve_params.index.max()}')
        print(f'Estimating yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        super().fit(**kwargs)
        
    def visualize(self, use_plotly = False, print_summary=False, start = None, end = None,  xlim = 30, ylim_lower = 270, ylim_upper = 350):
        if not self.initialized:
            print('Nothing to plot, yield curve not initialized.')
        
        return plot_curves(self.yield_curves, 
                    daily = True, 
                    use_plotly = use_plotly, 
                    print_summary=print_summary, 
                    start = start, 
                    end = end, 
                    xlim = 30, 
                    ylim_lower = 270, 
                    ylim_upper = 350,
                           return_figure=return_figure)

class RealTimeYieldCurveBase(YieldCurve):
    def __init__(self, **kwargs):
        super().__init__( **kwargs) 
        self.curve_type = 'REALTIME'
        
    def initialize_data(self, **kwargs):
        if kwargs.get('use_redis'): use_redis = True
        else: use_redis = False
        
        print(f'Loading real time yield curve values from {((1-use_redis)*"BigQuery" + use_redis *"Redis")}')
        self.yield_curve_params = load_yield_curve_params(daily=False, 
                                                          date_start= self.start_date.strftime('%Y-%m-%d'), 
                                                          date_end = self.end_date.strftime('%Y-%m-%d'), 
                                                          use_redis=use_redis)
        self.initialized = True
        print(f'Real time yield curve values loaded, data spans {self.yield_curve_params.index.min()} to {self.yield_curve_params.index.max()}')
        print(f'Estimating yield curves for maturity T = {min(self.t)} to T = {max(self.t)}')
        super().fit(**kwargs)
        
        
    def visualize(self, date, use_plotly = False, print_summary=False, xlim = 30, ylim_lower = 270, ylim_upper = 350, return_figure=False):
        if not self.initialized:
            print('Nothing to plot, yield curve not initialized.')

        return plot_curves(self.yield_curves, 
                            daily = False,
                            date = date,
                            use_plotly = use_plotly, 
                            print_summary=print_summary, 
                            start = None, 
                            end = None, 
                            xlim = xlim, 
                            ylim_lower = ylim_lower, 
                            ylim_upper = ylim_upper,
                          return_figure=return_figure)

class RealTimeYieldCurve(RealTimeYieldCurveBase): 
    def __init__(self, **kwargs):
        super().__init__( **kwargs)         
        
    def fit(self, **kwargs):
        
        if 'model' in kwargs:
            self.model = kwargs.get('model')
        else:
            from sklearn.linear_model import Ridge
            self.model = Ridge()
            
        if 'scaler' in kwargs: 
            self.scaler = kwargs.get('scaler')
        else:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            
        if 'L_upper' in kwargs: 
            L_upper = kwargs.get('L_upper')
        else: 
            L_upper = 20 
        
        print('Loading index and maturity data from BigQuery')
        self.load_index_data()
        self.load_maturity_data()
        self.index_data = self.index_data.sort_index(ascending=True)
        self.maturity_data = self.maturity_data.sort_index(ascending=True)
        
        yield_curve_params = pd.DataFrame(columns=['const', 'exponential', 'laguerre', 'exponential_mean', 'exponential_std', 'laguerre_mean', 'laguerre_std'])
        
        target_dates = [x.strftime('%Y-%m-%d') for x in set(self.index_data.index.intersection(self.maturity_data.index))] # get common dates between index and maturity, drop duplicates
        print(f'Optimizing curves for {min(target_dates)} to {max(target_dates)}, for shape parameter up to {L_upper}')
        
        optimization_results = pd.DataFrame(columns = ['L', 'MAE', 'model', 'scaler'])
        self.results = {}
        
        for target_date in target_dates:
            maturity_dict = get_maturity_dict(self.maturity_data, target_date)
            summary_df = pd.DataFrame(self.index_data.loc[target_date])
            summary_df.index
            summary_df.columns = ['ytw']
            summary_df['Weighted_Maturity'] = summary_df.index.map(maturity_dict).astype(float)

            
            result_df = pd.DataFrame(columns=['L','MAE','model', 'scaler'])
            
            for L in range(1, L_upper+1):
                model, mae, scaler = run_NL_model(Ridge(alpha=0.01), 
                                                  StandardScaler(), 
                                                  summary_df, 
                                                  L)
                
                result_df = result_df.append({'L':L, 
                                              'MAE':mae, 
                                              'model':model, 
                                              'scaler': scaler},
                                             ignore_index=True)

            best_L, _, best_model, best_scaler = result_df.sort_values(by='MAE', ascending=True).iloc[0, :].values
            self.results[target_date] = result_df
            optimization_results = optimization_results.append(result_df.sort_values(by='MAE', ascending=True).iloc[0, :],
                                                              ignore_index=True)
            
            const = best_model.intercept_
            exponential, laguerre = best_model.coef_
            exponential_mean, laguerre_mean = best_scaler.mean_
            exponential_std, laguerre_std = np.sqrt(best_scaler.var_)

            yield_curve_params = yield_curve_params.append({'const':const, 
                                                  'exponential':exponential, 
                                                  'laguerre':laguerre, 
                                                  'exponential_mean':exponential_mean, 
                                                  'exponential_std':exponential_std, 
                                                  'laguerre_mean':laguerre_mean, 
                                                  'laguerre_std':laguerre_std,
                                                            'L' :best_L}, 
                                                 ignore_index=True)

        yield_curve_params.index = target_dates

        self.optimization_results = result_df
        self.yield_curve_params = yield_curve_params
        
        temp = self.yield_curve_params.apply(lambda x: predict_ytw(self.t, 
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
        temp.index = self.yield_curve_params.index.values
        temp.columns = self.t.flatten()
        self.yield_curves = temp
        self.initialized = True
