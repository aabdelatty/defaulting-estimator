from pydantic import BaseModel
from typing import Optional

class InputSchema(BaseModel):

    uuid: str
    account_amount_added_12_24m: int
    account_days_in_dc_12_24m: Optional[float] = None
    account_days_in_rem_12_24m: Optional[float] = None
    account_days_in_term_12_24m: Optional[float] = None
    account_incoming_debt_vs_paid_0_24m: Optional[float] = None
    account_status: Optional[float] = None
    account_worst_status_0_3m: Optional[float] = None
    account_worst_status_12_24m: Optional[float] = None
    account_worst_status_3_6m: Optional[float] = None
    account_worst_status_6_12m: Optional[float] = None
    age: int
    avg_payment_span_0_12m: Optional[float] = None
    avg_payment_span_0_3m: Optional[float] = None
    merchant_category: str
    merchant_group: str
    has_paid: bool
    max_paid_inv_0_12m: float
    max_paid_inv_0_24m: float
    name_in_email: str
    num_active_div_by_paid_inv_0_12m: Optional[float] = None
    num_active_inv: int
    num_arch_dc_0_12m: int
    num_arch_dc_12_24m: int
    num_arch_ok_0_12m: int
    num_arch_ok_12_24m: int
    num_arch_rem_0_12m: int
    num_arch_written_off_0_12m: Optional[float] = None
    num_arch_written_off_12_24m: Optional[float] = None
    num_unpaid_bills: int
    status_last_archived_0_24m: int
    status_2nd_last_archived_0_24m: int
    status_3rd_last_archived_0_24m: int
    status_max_archived_0_6_months: int
    status_max_archived_0_12_months: int
    status_max_archived_0_24_months: int
    recovery_debt: int
    sum_capital_paid_account_0_12m: int
    sum_capital_paid_account_12_24m: int
    sum_paid_inv_0_12m: int
    time_hours: float
    worst_status_active_inv: Optional[float] = None
