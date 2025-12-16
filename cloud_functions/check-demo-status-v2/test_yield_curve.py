'''
'''
from auxiliary_variables import X_AXIS_LABEL_FOR_YIELD_CURVE, Y_AXIS_LABEL_FOR_YIELD_CURVE, ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE
from auxiliary_functions import run_multiple_times_before_failing, response_from_yield_curve_plot, response_from_yield_curve_table


def _test_yield_curve_at_current_datetime(display_type):
    assert display_type in ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE, f'`display_type` is {display_type}, but must be in {ACCEPTABLE_DISPLAY_TYPES_FOR_YIELD_CURVE}'
    response_func = response_from_yield_curve_plot if display_type == 'plot' else response_from_yield_curve_table
    response_dict = response_func()
    assert X_AXIS_LABEL_FOR_YIELD_CURVE in response_dict, f'{X_AXIS_LABEL_FOR_YIELD_CURVE} not in `response_dict`'
    assert Y_AXIS_LABEL_FOR_YIELD_CURVE in response_dict, f'{Y_AXIS_LABEL_FOR_YIELD_CURVE} not in `response_dict`'


@run_multiple_times_before_failing
def test_yield_curve_plot_at_current_datetime():
    _test_yield_curve_at_current_datetime('plot')


@run_multiple_times_before_failing
def test_yield_curve_table_at_current_datetime():
    _test_yield_curve_at_current_datetime('plot')
