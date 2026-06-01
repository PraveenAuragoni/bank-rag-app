"""
Creates sample bank policy PDFs for testing.
Run once: python create_sample_pdfs.py
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


def create_pdf(filepath, sections):
    doc = SimpleDocTemplate(filepath, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    for title, content in sections:
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(content, styles["Normal"]))
        story.append(Spacer(1, 20))
    doc.build(story)
    print(f"Created: {filepath}")


def main():
    os.makedirs("bank_docs", exist_ok=True)

    create_pdf("bank_docs/savings_account.pdf", [
        ("Savings Account — Minimum Balance",
         "Metro cities minimum balance is INR 10,000. Rural and semi-urban areas "
         "minimum balance is INR 5,000. Failure to maintain attracts penalty of "
         "INR 200 per month. Penalty waived for senior citizens above 60 years."),
        ("Savings Account — Interest Rate",
         "Interest rate is 3.5% per annum on daily closing balance. "
         "Credited to account on quarterly basis. No TDS deducted for interest "
         "below INR 40,000 per year. Form 15G/15H can be submitted to avoid TDS."),
        ("Savings Account — Features",
         "Free NEFT and RTGS transactions via internet banking. "
         "5 free ATM transactions per month at any ATM in India. "
         "Free cheque book of 25 leaves per year. Nomination facility available."),
        ("Savings Account — Opening Requirements",
         "Minimum opening deposit INR 1,000. KYC documents mandatory. "
         "Joint account allowed with maximum 4 holders. "
         "Minor account allowed with guardian as joint holder."),
    ])

    create_pdf("bank_docs/fixed_deposit.pdf", [
        ("Fixed Deposit — Interest Rates",
         "3 months: 5.5% per annum. 6 months: 6.0% per annum. "
         "12 months: 6.75% per annum. 24 months: 7.0% per annum. "
         "36 months: 7.25% per annum. 60 months: 7.5% per annum."),
        ("Fixed Deposit — Senior Citizen Rates",
         "Senior citizens above 60 years get additional 0.25% on all tenures. "
         "Super senior citizens above 80 years get additional 0.50%. "
         "Senior citizen rates applicable only on deposits up to INR 2 crore."),
        ("Fixed Deposit — Premature Withdrawal",
         "Premature withdrawal allowed after minimum 7 days. "
         "Penalty of 1% below the applicable rate for the period held. "
         "No penalty for withdrawal after 5 years. Tax saver FD cannot be "
         "withdrawn before 5 years."),
        ("Fixed Deposit — Auto Renewal",
         "Auto renewal available at maturity at prevailing rates. "
         "Customer can opt out 7 days before maturity. "
         "Nomination facility available. Joint FD allowed."),
    ])

    create_pdf("bank_docs/home_loan.pdf", [
        ("Home Loan — Interest Rate",
         "Floating rate starts at 8.5% per annum linked to repo rate. "
         "Fixed rate option available at 9.5% per annum for first 3 years. "
         "Women borrowers get 0.05% concession on floating rate."),
        ("Home Loan — Eligibility",
         "Minimum age 21 years. Maximum age 65 years at loan maturity. "
         "Minimum salary INR 30,000 per month for salaried. "
         "Minimum ITR of INR 3,60,000 per annum for self employed. "
         "Credit score of 700 and above required."),
        ("Home Loan — Processing Fee",
         "Processing fee 0.5% of loan amount. Minimum INR 5,000. "
         "Maximum INR 25,000. GST applicable on processing fee. "
         "Processing fee fully refundable if loan rejected by bank. "
         "Not refundable if rejected by customer after sanction."),
        ("Home Loan — Maximum Tenure",
         "Maximum tenure 30 years. Minimum tenure 5 years. "
         "Tenure reduced if customer age exceeds 65 years at maturity. "
         "Part prepayment allowed without penalty on floating rate loans."),
    ])

    create_pdf("bank_docs/kyc.pdf", [
        ("KYC — Individual Documents",
         "Mandatory: Aadhaar Card original and self-attested copy. "
         "PAN Card mandatory for transactions above INR 50,000. "
         "Recent passport-size photograph. Address proof not older than 3 months "
         "such as utility bill, bank statement, or rent agreement."),
        ("KYC — NRI Documents",
         "NRI customers require: Passport copy with visa stamp. "
         "Overseas address proof such as utility bill or bank statement. "
         "NRE or NRO account opening form. FEMA declaration form. "
         "In-person verification at branch or video KYC available."),
        ("KYC — Video KYC",
         "Video KYC available Monday to Saturday 9AM to 6PM. "
         "Customer requires Aadhaar with linked mobile number for OTP. "
         "Live photograph captured during video call. "
         "Signature on white paper required during video call. "
         "Account activated within 24 hours of successful video KYC."),
        ("KYC — Re-KYC",
         "Re-KYC required every 2 years for high risk customers. "
         "Every 8 years for low risk customers. "
         "Failure to submit Re-KYC results in account restrictions. "
         "Can be submitted at branch or uploaded via internet banking."),
    ])

    create_pdf("bank_docs/digital_banking.pdf", [
        ("Internet Banking — Password Policy",
         "Password must be 8 to 16 characters. Must contain at least one uppercase "
         "letter, one lowercase letter, one number, and one special character. "
         "Password must be reset every 90 days. Last 5 passwords cannot be reused."),
        ("Internet Banking — Account Lockout",
         "Three consecutive wrong password attempts will lock the account for 24 hours. "
         "Account can be unlocked by calling customer care or visiting branch. "
         "Self-unlock available via registered mobile OTP after 2 hours."),
        ("Internet Banking — OTP Policy",
         "OTP valid for 10 minutes only. OTP sent to registered mobile number only. "
         "Never share OTP with anyone including bank staff. "
         "Bank will never ask for OTP over phone or email."),
        ("Mobile Banking — Transaction Limits",
         "Default daily transfer limit INR 1,00,000. Can be increased to INR 5,00,000 "
         "by visiting branch. UPI limit INR 1,00,000 per transaction. "
         "International transfers require separate SWIFT application."),
    ])

    create_pdf("bank_docs/transfers.pdf", [
        ("NEFT — Transfer Policy",
         "NEFT processes in batches every 30 minutes during banking hours. "
         "Available 8AM to 7PM Monday to Saturday. Not available on Sundays "
         "and bank holidays. No minimum or maximum amount limit. "
         "Free for retail customers via internet and mobile banking."),
        ("RTGS — Transfer Policy",
         "RTGS minimum amount INR 2,00,000. No maximum limit. "
         "Available 8AM to 6PM on weekdays. 8AM to 2PM on Saturdays. "
         "Settlement is real-time and final. Free for retail customers."),
        ("IMPS — Transfer Policy",
         "IMPS available 24 hours 7 days including holidays. "
         "Maximum INR 5,00,000 per transaction. "
         "Charges: INR 5 for up to INR 1,000. INR 15 for INR 1,001 to INR 1,00,000. "
         "INR 25 for above INR 1,00,000."),
        ("UPI — Transfer Policy",
         "UPI available 24 hours 7 days. Maximum INR 1,00,000 per transaction. "
         "No charges for P2P transfers. No charges for P2M up to INR 2,000. "
         "Linked to registered mobile number only."),
    ])

    print("\nAll sample PDFs created in bank_docs/ folder!")
    print("Now run: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
