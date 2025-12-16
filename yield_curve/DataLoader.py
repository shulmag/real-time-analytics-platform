DATA_QUERY = """
    SELECT
      ifnull(settlement_date,
        assumed_settlement_date) AS settle_date,
      maturity_date,
      coupon,
      is_non_transaction_based_compensation,
      dated_date,
      issue_price,
      interest_payment_frequency,
      par_call_date,
      next_call_price,
      par_call_price,
      previous_coupon_payment_date,
      next_coupon_payment_date,
      first_coupon_date,
      coupon_type,
      muni_security_type,
      called_redemption_type,
      refund_price,
      call_timing,
      next_call_date AS call_date,
      next_sink_date AS sink_date,
      delivery_date AS deliv_date,
      refund_date AS refund_date,
      max_schedule AS sched_date,
      organization_primary_name AS issuer,
      is_callable,
      is_called,
      sp_long AS sp_lt_rating,
      advanced_assumed_maturity_date,
      recent
    FROM
      `eng-reactor-287421.primary_views.trade_history_with_reference_data`
    WHERE
      yield IS NOT NULL
      AND par_traded IS NOT NULL
      AND sp_long IS NOT NULL
      AND sp_long != "NR"
      AND federal_tax_status = 2
      -- AND trade_date >= '2021-01-01'
    ORDER BY
      maturity_date ASC
      
    LIMIT 10
  """

class DataLoader(object):
    '''
    Class to load the data from big query 
    and process it to create train and test data
    when calling the class, the user needs to provide
    the big query client and the query to fetch data
     '''
    def __init__(self,query,client):
        self.query = query
        self.trade_dataframe = None
        self.client = client
        self.trade_features = ['rtrs_control_number',
                               'trade_datetime',
                               'publish_datetime',
                               'msrb_valid_from_date',
                               'msrb_valid_to_date',
                               'yield_spread',
                               'yield',
                               'dollar_price',
                               'par_traded',
                               'trade_type',
                               'seconds_ago']
        

    @staticmethod
    def tradeDictToList(trade_dict: dict) -> list:
        '''
        This function converts the recent trades dictionary
        to a list. 

        The SQL arrays are converted to a dictionary
        when read as a pandas dataframe and need to processed
        before they can be fed into the ML model

        parameters:
        trade_dict : dict
        '''
       	
        return list(trade_dict.values())
    
    @staticmethod
    def tradeListToArray(trade_history):
        '''
        parameters:
        trade_history - list
        
        This function creates the trade history array
        '''
        if len(trade_history) == 0:
            return np.array([])
        
        trades_list = [DataLoader.tradeDictToList(entry) for entry in trade_history]
        return np.stack(trades_list)
    
        
    def fetchData(self):
        '''
        Function executes the query to fetch data from BigQuery
        and apply helper functions to create the trade history array
        '''               
        #if os.path.isfile('transformer.pkl'):
        #    self.trade_dataframe = pd.read_pickle('transformer.pkl')
        #else:
        self.trade_dataframe = self.client.query(self.query).result().to_dataframe()
        self.trade_dataframe['trade_history'] = self.trade_dataframe.recent.apply(self.tradeListToArray)
        self.trade_dataframe.drop(columns=['recent'],inplace=True)
        
    # Class functions do not need an instance of the class to be called.
    # They are methods associated with the class and not the instance
    # and can be called by the class directly
    @classmethod
    def processData(cls,query,client):
        '''
        Class method to process data from bigquery to create
        a dataframe that can be used to train the ML models
        
        The 
        '''
        instance = cls(query,client)
        instance.fetchData()
        instance.trade_dataframe = instance.trade_dataframe.explode("trade_history",ignore_index=True)
        
        temp_df = pd.DataFrame()
        temp_df[instance.trade_features] = pd.DataFrame(instance.trade_dataframe.trade_history.tolist(), index= instance.trade_dataframe.index)

        instance.trade_dataframe = instance.trade_dataframe.drop(columns=['trade_history'])
        instance.trade_dataframe = pd.concat([instance.trade_dataframe,temp_df], axis=1)
        instance.trade_dataframe.dropna(inplace=True)        
        display(instance.trade_dataframe.head())
        
        return instance.trade_dataframe
    
#%time rawdf = DataLoader.processData(DATA_QUERY,bq_client)