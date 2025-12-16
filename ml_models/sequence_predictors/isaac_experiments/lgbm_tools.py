from sklearn.ensemble import VotingRegressor
import lightgbm
from lightgbm import LGBMRegressor
from ficc_keras_utils import *

defaultdepth = 8
defaultseed = 881
defaultloss = 'mae' # 'huber' # 'fair'
huberdelta = 30
default_n_jobs = 1

def gbmprep(df, only=[], plus=[], minus=[]):
    if only != []: 
        USECOLS = only
    else:
        USECOLS = df.select_dtypes(include=['category','bool','number']).columns
    USECOLS = ( set(USECOLS) - mkleakers(TARGETS) - TOOMANY | set(plus) ) - set(minus)
    return df[ list(USECOLS) ]


def myLGBM(seed = defaultseed, depth = defaultdepth, loss = defaultloss):
    return LGBMRegressor(max_depth=depth, num_leaves=depth*10,  objective=loss, verbosity=-1, alpha = huberdelta,
                         n_estimators=depth*30, subsample = 0.5, subsample_freq = 10, # linear_tree = True,
                         random_state = seed, n_jobs = default_n_jobs, device_type = "cpu") 

def mkensemble(n = 4, seed = defaultseed, depth = defaultdepth, loss = defaultloss):
    regressors = []
    for j in range(0,n):
        regressors = regressors + [( 'm'+str(j), myLGBM(seed+j, depth, loss) )]
    return VotingRegressor( regressors, n_jobs = default_n_jobs, verbose = False )

# def trtestrows(df,target):
#     global first_train_date, first_test_date
#     train_rows = (df.trade_date >= first_train_date) & (df.trade_date < first_test_date) & df[target].notnull()
#     test_rows = (df.trade_date >= first_test_date) & df[target].notnull()
#     return train_rows, test_rows
    
def ess(weights):
    return weights.sum()**2 / (weights**2).sum()

def drawpoints(df, rows=[], label='new_ys', showplot=5):
    if len(rows) > 0: df = df[rows]
    label_preds = label + "_preds"
    label_ae = label + "_ae"

    df = truemid(df)
    
    if label   == 'new_ys':       ytw_preds = df[label_preds] + df.new_ficc_ycl
    elif label == 'diff_ys':      ytw_preds = df[label_preds] + df.new_ficc_ycl + df.last_yield_spread
    elif label == 'yield_spread': ytw_preds = df[label_preds] + df.ficc_ycl
    elif label == 'ytw':          ytw_preds = df[label_preds]
    else: ytw_preds = df.ytw
    
    ytw_ae = (df.ytw - ytw_preds).abs().mean()
    
    r, g, b = colors.to_rgb('red')
    w = np.minimum( np.array(10**df.quantity), threshold)
    opacity = w / threshold
    color = [(r, g, b, alpha) for alpha in opacity]

    if showplot > 0:
        plt.figure(figsize=(8,8))
        plt.scatter(df[label], df[label_preds], s=5, c=color)

    da = df[label_ae]
    n = len(da)
    print( f"\nLarge {directions} n={n}\t {label} MAE = {da.mean():.2f} +/- "\
          + f"{da.std():.2f} ({da.std()/np.sqrt(n):.2f}) median {da.median():.2f}     YTW MAE = {ytw_ae:.2f}" )
    sortby = label_ae

    top = df.sort_values(by=[sortby], ascending=False).iloc[:100,:]
    top = top.drop_duplicates(['issuer'])

    for (bond, d, x, y, lastytw, ytw, err, lastdp, dp, side, lastside, days, lastys, back, lastsize, nowsize, n) in \
                zip( top.cusip, top.trade_datetime, top[label], top[label_preds], top.last_ytw, top.ytw, top[sortby], \
                     top.last_dollar_price, top.dollar_price, top.trade_type, top.last_trade_type, top.last_duration, \
                     top.last_yield_spread, top.last_seconds_ago, top.last_size, top.par_traded, range(100) ):
        if n >= showplot: break
        try:
            seconds_ago = 10**back
            if seconds_ago > 43200:
                ago = str(int(np.ceil(seconds_ago / 86400))) + "d"
            else:
                ago = str(int(np.floor(seconds_ago / 60))) + "min"
        except Exception as e: 
            ago = "***"   
            
        days *= 360   # last_duration is in fractional years
        print( f"{bond:9} {d:%m-%d}  back {ago:>6}: {10**lastsize/1000:5.0f}K {lastside:1}{lastytw:4.0f} ${lastdp:6.2f} dur {days:5.0f} " \
              + f"ys {lastys:4.0f}  now: {nowsize/1000:5.0f}K {side:1}{ytw:4.0f} ${dp:6.2f}  true {x:7.2f} {y:7.2f} diff {x-y:7.2f}" )
                
        if n <= 10:
            plt.scatter(x,y, s=40, c = 'green')
            if (x < y): 
                plt.annotate(bond, (x-130,y-2.5))
            else:
                plt.annotate(bond, (x+10,y-2.5))
    print("\n")
    return

def traintest(PREDICTORS, train_dataframe, test_dataframe, target, n = 4, seed = defaultseed, depth = defaultdepth, loss = defaultloss, sample_weight=None,
              only=[], plus=[], minus=[], showplot=5, evaltrain=False):
    # global TARGETS
    # TARGETS = TARGETS | { target }
    
    # train_rows, test_rows = trtestrows(df,target)

    ens = mkensemble(n, seed, depth, loss)
    if n == 1:
        text = "Training one model "
    else:
        text = f"Training {n} models" 
    print(text + f"with {len(PREDICTORS)} columns on {len(train_dataframe)} examples starting {train_start}")
    if sample_weight is not None:
        print( f"Weighted samples with effective sample size = {ess(sample_weight):8.0f}" )
    print(f"Evaluating {target} predictions on {len(test_dataframe)} examples starting {test_start}")
    ens.fit( train_dataframe[PREDICTORS], train_dataframe[target], sample_weight )

    if evaltrain:
        maeval(PREDICTORS, train_dataframe, ens, target, only, plus, minus)
    maeval(PREDICTORS, test_dataframe, ens, target, only, plus, minus)
    # drawpoints(test_dataframe, target, showplot)     
    return ens

def extend(df, name, vals):
    # assert len(df) == len(rows), "*** len(df) != len(rows)"
    # if len(vals) == len(df): 
    #     vals = vals[rows]
    # assert len(vals) == np.sum(rows), "*** len(vals) != np.sum(rows)"
    
    if not name in df.columns: df[name] = 0.0
    df.loc[:, name] = vals
    return

def maeval(PREDICTORS, df, model, label, only=[], plus=[], minus=[]):
    dfp = model.predict(df[PREDICTORS])
    extend(df, label + "_preds",  dfp )

    delta = df[label] - dfp
    extend(df, label + "_err", delta )
    
    da = delta.abs()
    extend(df, label + "_ae", da )

    n = len(da)
    base = f"\n{label} n = {n}  bias = {delta.mean():5.2f}  MAE={da.mean():5.2f} +/- {da.std():.2f}"
    print( base + f" ({da.std()/np.sqrt(n):.2f})\t median {da.median():5.2f}" )
    return df

def myplotimportance(model, nfeatures = 30):
    if isinstance( model, LGBMRegressor ): 
        m = model
    elif isinstance( model.estimators_[0], LGBMRegressor ): 
        m = model.estimators_[-1]
    else: 
        print("*** not a valid model")
    lightgbm.plot_importance(m, importance_type="gain", precision=0, ignore_zero=False,\
                             max_num_features=nfeatures, figsize=(8,np.ceil(nfeatures/7)) )
    
def imptdf(model):
    m = model.estimators_[-1]
    imps = pd.DataFrame([m.feature_name_, m.booster_.feature_importance(importance_type='gain'), m.feature_importances_]).T
    imps.columns = ['name', 'gain', 'splits']
    imps = imps.sort_values(by='gain', ascending=False)
    # imps['gain'] = (imps.gain/imps.gain.mean()) / (imps.splits/imps.splits.mean())
    imps['gain'] = imps.gain/1e6
    return imps


def myplotimportance(model, nfeatures = 30):
    if isinstance( model, LGBMRegressor ): 
        m = model
    elif isinstance( model.estimators_[0], LGBMRegressor ): 
        m = model.estimators_[-1]
    else: 
        print("*** not a valid model")
    lightgbm.plot_importance(m, importance_type="gain", precision=0, ignore_zero=False,\
                             max_num_features=nfeatures, figsize=(8,np.ceil(nfeatures/7)) )