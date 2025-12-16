TRADE_SEQUENCE_LENGTH = None
NUM_FEATURES = None
LEARNING_RATE = None
VALIDATION_SPLIT = None
BATCH_SIZE = None
NUM_EPOCHS = None
DROPOUT = None

BINARY_FEATURES = ['callable',
          'sinking',
          'zerocoupon',
          'is_non_transaction_based_compensation',
          'is_general_obligation',
          'callable_at_cav',
          'extraordinary_make_whole_call',
          'make_whole_call',
          'has_unexpired_lines_of_credit',
          'escrow_exists']

CATEGORICAL_FEATURES = ['rating',
                        'incorporated_state_code',
                        'trade_type',
                        'purpose_class',
                        'max_ys_ttypes',
                        'min_ys_ttypes',
                        'max_qty_ttypes',
                        'min_ago_ttypes',
                        'D_min_ago_ttypes',
                        'P_min_ago_ttypes',
                        'S_min_ago_ttypes']

NON_CATEGORICAL_FEATURES = ['quantity',
                     'days_to_maturity',
                     'days_to_call',
                     'coupon',
                     'issue_amount',
                     'last_seconds_ago',
                     'last_yield_spread',
                     'days_to_settle',
                     'days_to_par',
                     'maturity_amount',
                     'issue_price',
                     'orig_principal_amount',
                     'max_amount_outstanding',
                     'accrued_days',
                     'days_in_interest_payment',
                     'A/E',
                     'ficc_treasury_spread',
                     'max_ys_ys',
                     'max_ys_ago',
                     'max_ys_qdiff',
                     'min_ys_ys',
                     'min_ys_ago',
                     'min_ys_qdiff',
                     'max_qty_ys',
                     'max_qty_ago',
                     'max_qty_qdiff',
                     'min_ago_ys',
                     'min_ago_ago',
                     'min_ago_qdiff',
                     'D_min_ago_ys',
                     'D_min_ago_ago',
                     'D_min_ago_qdiff',
                     'P_min_ago_ys',
                     'P_min_ago_ago',
                     'P_min_ago_qdiff',
                     'S_min_ago_ys',
                     'S_min_ago_ago',
                     'S_min_ago_qdiff']
                    
ADDITIONAL_SEQUENCES = []
