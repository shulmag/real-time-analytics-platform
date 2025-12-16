import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'../')

from yield_spread_model_mitas.train import train


'''This list was created from the feature importance list in a team member's 
notebook (CMD+F 'plot_importance'): https://github.com/Ficc-ai/ficc/blob/ficc_ml/ml_models/sequence_predictors/yield_spread_model_ahmad.ipynb'''
features_ordered_by_importance = ('last_yield_spread', 
                                  'incorporated_state_code', 
                                  'days_to_maturity', 
                                  'purpose_class', 
                                  'last_seconds_ago', 
                                  'accrued_days', 
                                  'quantity', 
                                  'issue_price', 
                                  'A/E', 
                                  'rating', 
                                  'days_to_call', 
                                  'issue_amount', 
                                  'trade_type', 
                                  'maturity_amount', 
                                  'coupon', 
                                  'days_to_par', 
                                  'orig_principal_amount', 
                                  'days_to_settle', 
                                  'is_non_transaction_based_compensation', 
                                  'purpose_sub_class', 
                                  'calc_day_cat', 
                                  'called_redemption_type', 
                                  'max_amount_outstanding', 
                                  'callable', 
                                  'make_whole_call', 
                                  'transaction_type', 
                                  'sinking', 
                                  'escrow_exists', 
                                  'is_general_obligation', 
                                  'purpose_sub_class', 
                                  'capital_type', 
                                  'use_of_proceeds', 
                                  'muni_issue_type', 
                                  'muni_security_type', 
                                  'state_tax_status', 
                                  'extraordinary_make_whole_call', 
                                  'other_enhancement_type', 
                                  'callable_at_cav', 
                                  'zerocoupon', 
                                  'asset_claim_code', 
                                  'orig_instrument_enhancement_type', 
                                  'sale_type', 
                                  'secured', 
                                  'has_unexpired_lines_of_credit', 
                                  'days_in_interest_payment', 
                                  'sec_regulation')


'''Add features one-by-one from `features_to_add` and train the `model_class` on 
`train_data`. Evaluate the model on `test_data`. We assume that both `test_data` 
and `train_data` have been encoded. We should observe that adding more features 
causes the test loss to decrease. If not, then additional features may be causing 
overfitting.'''
def monitor_performance_for_adding_features(model_class, train_data, test_data, batch_size, num_workers, label_encoders, binary_features, categorical_features, num_epochs, features_to_add):
    features_so_far = ['yield_spread']
    l1_test_losses = []
    l2_test_losses = []
    features_in_data = set(train_data.columns)
    features_not_found = []
    for feature in features_to_add:
        if feature not in features_in_data:
            features_not_found.append(feature)
            continue
        
        features_so_far.append(feature)
        losses, _ = train(model_class(batch_size, num_workers, train_data[features_so_far], test_data[features_so_far], label_encoders, binary_features, categorical_features), num_epochs)
        _, _, test_losses = losses
        l1_test_losses.append(test_losses[0])
        l2_test_losses.append(test_losses[1])
    print(f'The following features were not found: {features_not_found}')
    plt.plot(l1_test_losses, label='L1')
    plt.legend()
    plt.show()
    plt.plot(l2_test_losses, label='L2')
    plt.legend()
    plt.show()
    return l1_test_losses, l2_test_losses