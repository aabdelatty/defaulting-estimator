import typing as t

import numpy as np
import pandas as pd
from marshmallow import fields, Schema, ValidationError


class KlarnaDataInputSchema(Schema):

    uuid = fields.Str(allow_none=False)
    account_amount_added_12_24m = fields.Integer(allow_none=False)
    account_days_in_dc_12_24m = fields.Float(allow_none=True)
    account_days_in_rem_12_24m = fields.Float(allow_none=True)
    account_days_in_term_12_24m = fields.Float(allow_none=True)
    account_incoming_debt_vs_paid_0_24m = fields.Float(allow_none=True)
    account_status = fields.Float(allow_none=True)
    account_worst_status_0_3m = fields.Float(allow_none=True)
    account_worst_status_12_24m = fields.Float(allow_none=True)
    account_worst_status_3_6m = fields.Float(allow_none=True)
    account_worst_status_6_12m = fields.Float(allow_none=True)
    age = fields.Integer(allow_none=False)
    avg_payment_span_0_12m = fields.Float(allow_none=True)
    avg_payment_span_0_3m = fields.Float(allow_none=True)
    merchant_category = fields.Str(allow_none=False)
    merchant_group = fields.Str(allow_none=False)
    has_paid = fields.Boolean(allow_none=False)
    max_paid_inv_0_12m = fields.Float(allow_none=False)
    max_paid_inv_0_24m = fields.Float(allow_none=False)
    name_in_email = fields.Str(allow_none=False)
    num_active_div_by_paid_inv_0_12m = fields.Float(allow_none=True)
    num_active_inv = fields.Integer(allow_none=False)
    num_arch_dc_0_12m = fields.Integer(allow_none=False)
    num_arch_dc_12_24m = fields.Integer(allow_none=False)
    num_arch_ok_0_12m = fields.Integer(allow_none=False)
    num_arch_ok_12_24m = fields.Integer(allow_none=False)
    num_arch_rem_0_12m = fields.Integer(allow_none=False)
    num_arch_written_off_0_12m = fields.Float(allow_none=True)
    num_arch_written_off_12_24m = fields.Float(allow_none=True)
    num_unpaid_bills = fields.Integer(allow_none=False)
    status_last_archived_0_24m = fields.Integer(allow_none=False)
    status_2nd_last_archived_0_24m = fields.Integer(allow_none=False)
    status_3rd_last_archived_0_24m = fields.Integer(allow_none=False)
    status_max_archived_0_6_months = fields.Integer(allow_none=False)
    status_max_archived_0_12_months = fields.Integer(allow_none=False)
    status_max_archived_0_24_months = fields.Integer(allow_none=False)
    recovery_debt = fields.Integer(allow_none=False)
    sum_capital_paid_account_0_12m = fields.Integer(allow_none=False)
    sum_capital_paid_account_12_24m = fields.Integer(allow_none=False)
    sum_paid_inv_0_12m = fields.Integer(allow_none=False)
    time_hours = fields.Float(allow_none=False)
    worst_status_active_inv = fields.Float(allow_none=True)





def validate_inputs(
    *, input_data: pd.DataFrame
) -> t.Tuple[pd.DataFrame, t.Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # set many=True to allow passing in a list
    schema = KlarnaDataInputSchema(many=True)
    errors = None
    validated_data = input_data
    
    try:
        # replace numpy nans so that Marshmallow can validate
        schema.load(validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as exc:
        errors = exc.messages

    return validated_data, errors
