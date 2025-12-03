#!/bin/bash
mkdir -p /app/logs

# Start Ingestion Service
cd /app/ingestion_service
echo "Starting Ingestion Service on port 8011..." >> /app/logs/ingestion.log
uvicorn main:app --host 0.0.0.0 --port 8011 >> /app/logs/ingestion.log 2>&1 &

# Start Search Service
cd /app/search_service
echo "Starting Search Service on port 8012..." >> /app/logs/search.log
uvicorn main:app --host 0.0.0.0 --port 8012 >> /app/logs/search.log 2>&1 &

# Start Streamlit App
cd /app/streamlit_app
echo "Starting Streamlit App on port 8503..." >> /app/logs/app.log
streamlit run app.py --server.port=8503 --server.address=0.0.0.0 >> /app/logs/app.log 2>&1 &

# Wait for any process to exit
wait -n
exit $?
