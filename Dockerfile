FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data/templates ./data/templates
RUN pip install --no-cache-dir ".[dashboard,parquet]"

EXPOSE 8501
# Equity dashboard by default. For sports markets:
#   docker run --rm -p 8501:8501 doppler-garch streamlit run src/doppler_research/sports/dashboard.py --server.address=0.0.0.0
CMD ["streamlit", "run", "src/doppler_research/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
