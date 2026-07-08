

import pdfplumber
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("knowledge_base_raw")
OUT_DIR = Path("knowledge_base")
OUT_DIR.mkdir(exist_ok=True)

# Clean filename mapping — makes source citations readable
FILENAME_MAP = {
    "09. APIT_2526_Table_08_Text.pdf":          "APIT_Tax_Table_8_2025_26.txt",
    "11-2026_E.pdf":                             "Amendment_Act_11_2026.txt",
    "2023_2334-21_E.pdf":                        "Gazette_2023_WHT_Expansion.txt",
    "Asmt_IIT_004_2022_2023_E.pdf":              "Tax_Return_Guide_2022_23.txt",
    "Guide to income sources(Draft).pdf":        "Guide_Income_Sources_Draft.txt",
    "IRA_Cons_Act_-_2025_Changes.pdf":           "IRA_Consolidated_2025.txt",
    "IR_Act_No._04_2023_E.pdf":                  "IR_Amendment_Act_04_2023.txt",
    "IR_Act_No._45_2022_E.pdf":                  "IR_Amendment_Act_45_2022.txt",
    "IR_Act_No_24_2017_E.pdf":                   "IR_Act_24_2017_Original.txt",
    "SEC_PN_IT_2026-02_E.pdf":                   "IRD_Practice_Note_2026_02.txt",
    "SEC_PN_IT_2026-03.pdf":                     "IRD_Practice_Note_2026_03.txt",
    "The_Inland_Revenue_Amending_Act_June_2026.pdf": "IR_Amending_Act_June_2026.txt",
}


def extract_pdf(pdf_path: Path) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


def main():
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {RAW_DIR}/")
        print("Download IRD PDFs and place them there first.")
        return

    print(f"Found {len(pdfs)} PDFs — converting to .txt\n")
    success, failed = 0, []

    for pdf_path in tqdm(pdfs, desc="Converting"):
        out_name = FILENAME_MAP.get(pdf_path.name, pdf_path.stem + ".txt")
        out_path = OUT_DIR / out_name
        try:
            text = extract_pdf(pdf_path)
            if len(text.strip()) < 200:
                print(f"\n  {pdf_path.name} — very little text ({len(text)} chars). May be scanned.")
                failed.append(pdf_path.name)
                continue
            out_path.write_text(text, encoding="utf-8")
            print(f"\n  {pdf_path.name} → {out_name} ({len(text):,} chars)")
            success += 1
        except Exception as e:
            print(f"\n  {pdf_path.name} failed: {e}")
            failed.append(pdf_path.name)

    print(f"\n{'='*50}")
    print(f"Done: {success} converted, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    print(f"\nNext step: uv run python src/ingest.py")


if __name__ == "__main__":
    main()
