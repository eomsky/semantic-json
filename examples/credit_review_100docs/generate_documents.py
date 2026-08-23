from pathlib import Path

BASE = Path(__file__).parent / "documents"
BASE.mkdir(parents=True, exist_ok=True)

def generate_irrelevant_documents() -> None:
    """doc_003~doc_100 생성 (Generate reproducible irrelevant documents)."""
    for i in range(3,101):
        letter=chr(ord("D")+((i-3)%20))
        if i%2:
            text=f"{letter}기업은 최근 매출이 안정적으로 유지되고 있으며 주요 거래처 구성에도 큰 변화가 없다. 영업현금흐름은 정상 범위에서 유지되고 있다. 해당 기업의 차입구조는 장기 중심으로 구성되어 있다.\n"
        else:
            text=f"{letter} Corp. operates in a different industry. Revenue and profitability remained broadly stable. The company maintained adequate liquidity and has continued normal operations.\n"
        (BASE/f"doc_{i:03d}.txt").write_text(text,encoding="utf-8")

if __name__=="__main__":
    generate_irrelevant_documents()
    print("Generated doc_003.txt through doc_100.txt")
