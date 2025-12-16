'''
'''
DEFAULT_PRICING_ERROR_MESSAGE = 'Pricing error'
NO_TOKEN_ERROR_MESSAGE = 'You have been logged out due to a period of inactivity. Refresh the page!'


class CustomMessageError(Exception):
    '''Exception raised with an error message to be displayed in the frontend.'''
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

    def get_json_response(self):
        from flask import jsonify, make_response    # lazy loading for lower latency
        
        return make_response(jsonify({'error': self.message}), 200)    # if we do response code >= 400, then it automatically triggers error handling on the front end, but we want to perform our own manual error handling
