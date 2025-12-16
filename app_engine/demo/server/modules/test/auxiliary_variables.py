'''
'''
DIRECTORY = 'tests/files'
USERNAME = 'eng@ficc.ai'
PASSWORD = 'Apace3745'

QUANTITY = 500    # this is in thousands
TRADE_TYPE = 'S'    # default trade type is customer buy

LOGGED_OUT_MESSAGE = 'You have been logged out due to a period of inactivity. Refresh the page!'    # must be identical to `loggedOutMessage` in `src/components/pricing.jsx`


class LoggedOutError(Exception):
    '''Raised when a test fails because the user was logged out. The error message displayed is 
    'You have been logged out due to a period of inactivity. Refresh the page!'.'''
    def __init__(self):
        super().__init__(LOGGED_OUT_MESSAGE)
