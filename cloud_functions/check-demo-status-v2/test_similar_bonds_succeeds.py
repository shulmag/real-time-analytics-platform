'''
'''
from auxiliary_functions import run_multiple_times_before_failing, response_from_similar_bonds, check_if_response_from_similar_bonds_when_individually_priced_is_successful


@run_multiple_times_before_failing
def test_similar_bonds_64971XQM3():
    cusip = '64971XQM3'
    response_dict = check_if_response_from_similar_bonds_when_individually_priced_is_successful(cusip)
    assert len(response_dict) > 0, f'When searching for similar bonds to CUSIP {cusip}, there should be some bonds, but there were none'


@run_multiple_times_before_failing
def test_no_similar_bonds():
    '''Tests that there is no error in the response when there are no similar bonds found.'''
    response_dict = response_from_similar_bonds(state='RI', rating='BBB+', purposeClass=27)
    bond_description = 'BBB+ rated Rhode Island education bonds'
    assert 'error' not in response_dict, f'Searching for similar bonds to {bond_description} should not have an error, but has an error: {response_dict["error"]}'
    assert len(response_dict) == 0, f'When searching for similar bonds to {bond_description}, there should be none, but there are actually {len(response_dict)} trades that are similar'
