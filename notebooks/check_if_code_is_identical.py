code1 = '''def create_cusip_in_cusip_list_clause(cusip_list=None, table_identifier: str = '') -> str:
    if cusip_list is None: return ''
    if table_identifier is not None: table_identifier = table_identifier + '.'
    cusip_list_as_tuple_string = str(tuple(cusip_list)) if len(cusip_list) > 1 else f'("{cusip_list[0]}")'
    return f' AND {table_identifier}cusip IN {cusip_list_as_tuple_string}'    # use `tuple(...)` to have the string representation with parentheses instead of square brackets'''
code2 = '''def create_cusip_in_cusip_list_clause(cusip_list=None, table_identifier: str = '') -> str:
    if cusip_list is None: return ''
    if table_identifier is not None: table_identifier = table_identifier + '.'
    cusip_list_as_tuple_string = str(tuple(cusip_list)) if len(cusip_list) > 1 else f'("{cusip_list[0]}")'
    return f' AND {table_identifier}cusip IN {cusip_list_as_tuple_string}'    # use `tuple(...)` to have the string representation with parentheses instead of square brackets'''

print(code1 == code2)