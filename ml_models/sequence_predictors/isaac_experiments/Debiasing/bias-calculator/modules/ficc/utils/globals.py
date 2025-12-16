'''
                  the data processing module
 '''

def init():
    global FICC_ERROR 
    FICC_ERROR = ""
    
    global nelson_params
    nelson_params = None
    
    global scalar_params
    scalar_params = None

    global YIELD_CURVE_TO_USE
    YIELD_CURVE_TO_USE = "FICC"

    global mmd_ycl
    mmd_ycl = None

    global treasury_rate
    treasury_rate = None