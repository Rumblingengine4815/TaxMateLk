import json
from datetime import datetime

def calculate_wht(amount: float, is_service: bool = True) -> str:
    """
    Calculates the 5% Withholding Tax (WHT) on service fee payments.
    Based on the June 2026 Amendment Act.
    """
    if not is_service:
        return json.dumps({
            "status": "not_applicable",
            "message": "WHT applies specifically to service fee payments, not goods."
        })
        
    wht_amount = amount * 0.05
    net_amount = amount - wht_amount
    
    return json.dumps({
        "status": "success",
        "gross_amount": amount,
        "wht_deducted": wht_amount,
        "net_received": net_amount,
        "wht_rate": "5%",
        "message": f"A 5% WHT of LKR {wht_amount:,.2f} will be deducted. You will receive LKR {net_amount:,.2f}."
    })

def get_quarterly_schedule(year_of_assessment: str = None) -> str:
    """
    Returns the quarterly tax payment schedule for Sri Lanka.
    """
    if not year_of_assessment:
        current_year = datetime.now().year
        year_of_assessment = f"{current_year}/{current_year+1}"
        
    return json.dumps({
        "year_of_assessment": year_of_assessment,
        "installments": [
            {"quarter": 1, "due_date": "August 15"},
            {"quarter": 2, "due_date": "November 15"},
            {"quarter": 3, "due_date": "February 15"},
            {"quarter": 4, "due_date": "May 15"}
        ],
        "message": "Tax must be paid in 4 equal installments on or before the above dates."
    })

def calculate_tax(annual_income: float, has_foreign_exemption: bool = False) -> str:
    """
    Calculates the annual personal income tax for Sri Lankan freelancers.
    First 1,200,000 is exempt (Personal Relief).
    Following bands: 500k @ 6%, 12%, 18%, 24%, 30%, rest @ 36%.
    """
    if has_foreign_exemption:
        return json.dumps({
            "status": "exempt",
            "message": "If you are exporting services and receiving payment in foreign currency through the banking system, your income may be exempt or subject to a different regime. Please consult the specific foreign currency exemption rules."
        })

    taxable_income = max(0, annual_income - 1_200_000)
    tax_total = 0.0
    
    bands = [
        (500_000, 0.06),
        (500_000, 0.12),
        (500_000, 0.18),
        (500_000, 0.24),
        (500_000, 0.30)
    ]
    
    remaining = taxable_income
    band_breakdown = []
    
    for band_limit, rate in bands:
        if remaining > 0:
            amount_in_band = min(remaining, band_limit)
            tax_for_band = amount_in_band * rate
            tax_total += tax_for_band
            if amount_in_band > 0:
                band_breakdown.append(f"LKR {amount_in_band:,.2f} @ {rate*100:.0f}% = LKR {tax_for_band:,.2f}")
            remaining -= amount_in_band
            
    if remaining > 0:
        tax_for_band = remaining * 0.36
        tax_total += tax_for_band
        band_breakdown.append(f"LKR {remaining:,.2f} @ 36% = LKR {tax_for_band:,.2f}")
        
    return json.dumps({
        "status": "success",
        "gross_income": annual_income,
        "taxable_income": taxable_income,
        "total_tax_payable": tax_total,
        "effective_tax_rate": f"{(tax_total / annual_income * 100) if annual_income > 0 else 0:.1f}%",
        "band_breakdown": band_breakdown
    })

if __name__ == "__main__":
    print("Testing calculate_wht:")
    print(calculate_wht(150000))
    print("\nTesting get_quarterly_schedule:")
    print(get_quarterly_schedule("2026/2027"))
    print("\nTesting calculate_tax (LKR 4,000,000):")
    print(calculate_tax(4000000))
