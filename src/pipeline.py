from scraper import scrape_and_tag
from chunker import split_prose_and_code
from embedder import Embedder
from db import DBConnection

BMW_DOCUMENTS = [
    {
        "url": "https://www.bmw.co.id/content/dam/bmw/marketID/bmw_co_id/Brochures/pdf/Speccard_735iMSport.pdf.asset.1691640457119.pdf",
        "model": "735i M Sport",
        "model_year": "2023",
    },
    {
        "url": "https://www.bmw.co.id/content/dam/bmw/marketID/bmw_co_id/Brochures/pdf/2024/specsheet/BMW-Spec-card-web-updates-NIK24-The7.pdf.asset.1709810589522.pdf",
        "model": "735i M Sport",
        "model_year": "2024",
    },
    {
        "url": "https://www.bmw.co.id/content/dam/bmw/marketID/bmw_co_id/Brochures/pdf/2024/specsheet/735i-M-Sport-Spec-Card-2025.pdf.asset.1739958467497.pdf",
        "model": "735i M Sport",
        "model_year": "2025",
    },
    {
        # Main Website
        "url": "https://www.bmw.co.id/en/all-models/bmw-i/i7/bmw-i7-sedan.html",
        "model": "i7 xDrive60",
        "model_year": "2025",
    },
    {
        # Technical Data Website
        "url": "https://www.bmw.co.id/en/all-models/bmw-i/i7/bmw-i7-sedan-technical-data.html/bmw-i7-xdrive60-gran-lusso.bmw",
        "model": "i7 xDrive60",
        "model_year": "2025",
    },
    {
        "url": "https://www.bmw.co.id/content/dam/bmw/marketID/bmw_co_id/Brochures/pdf/spec-card-730Li-M-Sport-20220121.pdf.asset.1646109041258.pdf",
        "model": "730Li M Sport",
        "model_year": "2022",
    },
    {
        "url": "https://www.bmw.co.id/content/dam/bmw/marketID/bmw_co_id/Brochures/pdf/specification-card–bmw-7-Opulence-2021.pdf.asset.1634545632157.pdf",
        "model": "740Li Opulence",
        "model_year": "2021",
    },
    {
        "url": "https://www.oto.com/en/mobil-baru/bmw/7-series-sedan/740li-pure-excellence",
        "model": "740Li Pure Excellence",
        "model_year": "2015",
    }
    
]


def run_ingestion(
    url: str,
    manufacturer: str,
    model: str,
    model_year: str,
    embedder,
    db,
):
    print(f"--- Ingesting {manufacturer} {model} ({model_year}) ---")
    scraped_data = scrape_and_tag(url, manufacturer, model, model_year)

    chunks = split_prose_and_code(
        scraped_data["raw_markdown"],
        scraped_data["metadata"],
    )
    print(f"Generated {len(chunks)} context-preserved chunks.")

    for chunk in chunks:
        vector = embedder.embed(chunk["content"])
        metadata = chunk["metadata"]

        db.insert_chunk(
            manufacturer=metadata["manufacturer"],
            model=metadata["model"],
            model_year=metadata["model_year"],
            chunk_type=metadata["type"],
            content=chunk["content"],
            metadata=metadata,
            embedding=vector,
        )


if __name__ == "__main__":
    embedder = Embedder()
    db = DBConnection()

    print("Clearing database to prevent duplicates...")
    with db.conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE document_chunks;")

    for document in BMW_DOCUMENTS:
        run_ingestion(
            url=document["url"],
            manufacturer="BMW",
            model=document["model"],
            model_year=document["model_year"],
            embedder=embedder,
            db=db,
        )

    print("--- BMW 7 Series Indonesia Ingestion Complete! ---")
