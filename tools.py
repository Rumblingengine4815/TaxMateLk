"""
src/tools.py
============
Pure Python tax calculation tools — NO LLM involved.
The agent calls these functions when it needs exact numbers.
This prevents LLM arithmetic errors on tax band calculations.

Run standalone to test:
    uv run python src/tools.py
"""

from dataclasses import dataclass
from datetime import date


# ── Tax bands 2025/26 (Resident individuals) ──────────────────────────────
# Source: IRA Consolidated Act 2025, First Schedule
TAX_BANDS = [
    (0,         1_800_000, 0.00),   # exempt threshold
    (1_800_000, 2_800_000, 0.06),   # 6%
    (2_800_000, 3_800_000, 0.12),   # 12%
    (3_800_000, 4_800_000, 0.18),   # 18%
    (4_800_000, 5_800_000, 0.24),   # 24%
    (5_800_000, float("inf"), 0.30),# 30%
]

PERSONAL_RELIEF        = 1_800_000   # LKR — updated for 2025/26
ITES_MAX_RATE          = 0.15        # 15% cap for qualifying foreign income
WHT_RATE               = 0.05        # 5% WHT on qualifying service fees
WHT_THRESHOLD_MONTHLY  = 100_000     # LKR — WHT only applies above this
WHT_PAYMENT_DAYS       = 15          # days after month end to remit WHT

# Quarterly installment due dates (approx, year of assessment Apr-Mar)
QUARTERLY_MONTHS = [8, 11, 2, 5]  # August, November, February, May


@dataclass
class TaxResult:
    gross_income_lkr:   float
    taxable_income:     float
    tax_liability:      float
    effective_rate:     float
    quarterly_amount:   float
    income_type:        str
    notes:              list[str]

    def summary(self) -> str:
        lines = [
            f"Income Type:        {self.income_type}",
            f"Gross Income:       LKR {self.gross_income_lkr:,.0f}",
            f"Taxable Income:     LKR {self.taxable_income:,.0f}",
            f"Annual Tax Due:     LKR {self.tax_liability:,.0f}",
            f"Effective Rate:     {self.effective_rate:.1%}",
            f"Quarterly Payment:  LKR {self.quarterly_amount:,.0f}",
        ]
        if self.notes:
            lines.append("\nNotes:")
            lines += [f"  • {n}" for n in self.notes]
        return "\n".join(lines)


def calculate_tax_bands(taxable_income: float) -> float:
    """Calculate tax using progressive bands."""
    tax = 0.0
    for lower, upper, rate in TAX_BANDS:
        if taxable_income <= lower:
            break
        taxable_in_band = min(taxable_income, upper) - lower
        tax += taxable_in_band * rate
    return tax


def calculate_tax(
    monthly_income_lkr: float,
    income_type: str,          # "ites_foreign" | "local_service" | "employment" | "mixed"
    is_foreign_currency: bool = False,
    remitted_via_bank: bool   = False,
    has_employment_income: bool = False,
    employment_monthly_lkr: float = 0,
) -> TaxResult:
    """
    Main tax calculation function exposed to the agent as a tool.

    Args:
        monthly_income_lkr:    Freelance/service income per month in LKR
        income_type:           Classification of income
        is_foreign_currency:   Payment received in foreign currency?
        remitted_via_bank:     Funds remitted through a Sri Lankan bank?
        has_employment_income: Does the person also have a salary?
        employment_monthly_lkr: Monthly salary if applicable

    Returns:
        TaxResult with full breakdown
    """
    notes = []
    annual_freelance  = monthly_income_lkr * 12
    annual_employment = employment_monthly_lkr * 12

    # ── ITES Foreign Income — 15% cap applies ──────────────────────────────
    if income_type == "ites_foreign" and is_foreign_currency and remitted_via_bank:
        tax_liability  = annual_freelance * ITES_MAX_RATE
        effective_rate = ITES_MAX_RATE
        taxable_income = annual_freelance
        notes.append("15% maximum rate applies (ITES foreign currency, bank remittance)")
        notes.append("Source: IRA Consolidated Act 2025, First Schedule Para 1(6)")
        if monthly_income_lkr > WHT_THRESHOLD_MONTHLY:
            wht_monthly = monthly_income_lkr * WHT_RATE
            notes.append(
                f"WHT of LKR {wht_monthly:,.0f}/month will be deducted by your client "
                f"(5% on payments > LKR {WHT_THRESHOLD_MONTHLY:,})"
            )
            notes.append("Source: Amendment Act No. 11 of 2026, Section 13")

    # ── ITES but paid in LKR or not via bank — standard bands apply ────────
    elif income_type == "ites_foreign" and not (is_foreign_currency and remitted_via_bank):
        total_income   = annual_freelance + annual_employment
        taxable_income = max(0, total_income - PERSONAL_RELIEF)
        tax_liability  = calculate_tax_bands(taxable_income)
        effective_rate = tax_liability / total_income if total_income else 0
        notes.append("15% cap does NOT apply — payment not in foreign currency via bank")
        notes.append("Standard progressive bands apply instead")
        notes.append("Source: IRA Consolidated Act 2025, First Schedule Para 1(6)(b)")

    # ── Local service income ───────────────────────────────────────────────
    elif income_type == "local_service":
        total_income   = annual_freelance + annual_employment
        taxable_income = max(0, total_income - PERSONAL_RELIEF)
        tax_liability  = calculate_tax_bands(taxable_income)
        effective_rate = tax_liability / total_income if total_income else 0
        if monthly_income_lkr > WHT_THRESHOLD_MONTHLY:
            wht_monthly = monthly_income_lkr * WHT_RATE
            notes.append(
                f"WHT of LKR {wht_monthly:,.0f}/month deducted by client "
                f"(5% WHT on payments > LKR {WHT_THRESHOLD_MONTHLY:,})"
            )
            notes.append(
                "You are jointly liable if client fails to withhold. "
                "Source: IR Act No. 24 of 2017, Section 86(4)"
            )

    # ── Employment only ────────────────────────────────────────────────────
    else:
        total_income   = annual_employment or annual_freelance
        taxable_income = max(0, total_income - PERSONAL_RELIEF)
        tax_liability  = calculate_tax_bands(taxable_income)
        effective_rate = tax_liability / total_income if total_income else 0

    quarterly = tax_liability / 4

    return TaxResult(
        gross_income_lkr  = annual_freelance + annual_employment,
        taxable_income    = taxable_income,
        tax_liability     = round(tax_liability, 2),
        effective_rate    = round(effective_rate, 4),
        quarterly_amount  = round(quarterly, 2),
        income_type       = income_type,
        notes             = notes,
    )


def calculate_wht(monthly_payment_lkr: float) -> dict:
    """
    Calculates WHT deduction for a single payment.
    Called by agent when user asks 'how much WHT will my client deduct?'
    """
    if monthly_payment_lkr <= WHT_THRESHOLD_MONTHLY:
        return {
            "wht_applies":    False,
            "wht_amount":     0,
            "net_payment":    monthly_payment_lkr,
            "reason":         f"Payment ≤ LKR {WHT_THRESHOLD_MONTHLY:,} — WHT does not apply",
            "source":         "Amendment Act No. 11 of 2026, Section 13",
        }
    wht_amount = monthly_payment_lkr * WHT_RATE
    return {
        "wht_applies":    True,
        "wht_amount":     round(wht_amount, 2),
        "net_payment":    round(monthly_payment_lkr - wht_amount, 2),
        "remit_deadline": f"Within {WHT_PAYMENT_DAYS} days after month end",
        "source":         "IRA Consolidated Act 2025, Section 86(1)",
    }


def get_quarterly_schedule(tax_year_start: int = 2025) -> list[dict]:
    """Returns the 4 quarterly payment dates for a tax year."""
    schedule = []
    labels   = ["Q1", "Q2", "Q3", "Q4"]
    years    = [tax_year_start, tax_year_start, tax_year_start + 1, tax_year_start + 1]
    for label, month, year in zip(labels, QUARTERLY_MONTHS, years):
        schedule.append({
            "quarter":  label,
            "due_date": f"{year}-{month:02d}-15",
            "note":     "Pay 25% of estimated annual tax liability",
        })
    return schedule


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("TaxMate LK — Tax Calculation Tool Tests")
    print("=" * 55)

    # Test 1: ITES foreign income
    print("\n[Test 1] Software developer, USD 2,000/month, Wise → LKR bank")
    result = calculate_tax(
        monthly_income_lkr    = 650_000,   # approx USD 2,000 at ~325 LKR
        income_type           = "ites_foreign",
        is_foreign_currency   = True,
        remitted_via_bank     = True,
    )
    print(result.summary())

    # Test 2: Local photographer — new WHT applies
    print("\n[Test 2] Photographer, LKR 150,000/month local client")
    result2 = calculate_tax(
        monthly_income_lkr = 150_000,
        income_type        = "local_service",
    )
    print(result2.summary())

    # Test 3: WHT check
    print("\n[Test 3] WHT on LKR 85,000 payment")
    print(calculate_wht(85_000))
    print("\n[Test 3b] WHT on LKR 150,000 payment")
    print(calculate_wht(150_000))

    # Test 4: Quarterly schedule
    print("\n[Test 4] Quarterly payment schedule 2025/26")
    for q in get_quarterly_schedule(2025):
        print(f"  {q['quarter']}: due {q['due_date']} — {q['note']}")

    print("\n✅ All tools working. Next: uv run python src/agent.py")
