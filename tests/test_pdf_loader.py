from app.rag.ingest.pdf_loader import load_pdf

pages = load_pdf('datasets/sample_charts/sample_chart.pdf')
print(f'Total pages extracted: {len(pages)}')
for p in pages:
    print(f'--- Page {p["page"]} ---')
    print(p['text'][:300])
    print()