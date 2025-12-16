'''
'''
import time
import requests

from modules.test.auxiliary_variables import USERNAME, PASSWORD, LOGGED_OUT_MESSAGE, LoggedOutError
from modules.test.auxiliary_functions import REQUEST_URL, get_spreadsheet_as_list


def test_10_cusips_both_sides_1M(return_execution_time: bool = False):
    '''So far, this test just ensures that we get to the final line of the function.'''
    data = {'username': USERNAME, 'password': PASSWORD}
    cusip_list = ['562578PS7', '562578PS7', '078027CE7', '078027CE7', '656066QH3',
                  '656066QH3', '79771HAK9', '79771HAK9', '686356TE6', '686356TE6',
                  '015303DR8', '015303DR8', '527156CY7', '527156CY7', '473214JA3',
                  '473214JA3', '161285RC7', '161285RC7', '713253CT7', '713253CT7']
    data['cusipList'] = cusip_list
    data['quantityList'] = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
                            1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    data['tradeTypeList'] = ['S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P', 'S', 'P']

    # provide InvestorTools specific arguments to the API call
    data['ignoreErrorCusips'] = True
    data['setQuantityToAmountOutstandingIfLessThanGivenQuantity'] = True
    
    request_ref = f'{REQUEST_URL}/api/batchpricing'
    start_time = time.time()
    request_obj = requests.post(request_ref, data=data)
    time_elapsed = round(time.time() - start_time, 3)    # round the value in order to get a readable error statement

    # return_values = (request_obj, time_elapsed) if return_execution_time else return_values
    try:
        response_dict = request_obj.json()
    except Exception:
        pass
    if 'error' in response_dict and response_dict['error'] == LOGGED_OUT_MESSAGE:
        raise LoggedOutError

    content = get_spreadsheet_as_list(request_obj, spreadsheet_returned_as_json_string=True)
    print(content)
    assert True
