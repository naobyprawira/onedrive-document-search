import os
import requests
import streamlit as st
import logging

# Configure logging
os.makedirs("/app/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/app/logs/app.log")
    ]
)
logger = logging.getLogger("streamlit_app")

# Configuration
# Updated to use the new search service port (8012)
SEARCH_API_URL = os.getenv("SEARCH_API_URL", "http://localhost:8012/search")

st.set_page_config(
    page_title="Document Search",
    page_icon="🔍",
    layout="wide",
)

# Logo and Title
col1, col2 = st.columns([1, 6])
with col1:
    try:
        st.image("logo.png", width=80)
    except Exception:
        st.write("🔍")
with col2:
    st.title("Accounting Document Search")

# Removed description to directly show search input

# Authentication
APP_ACCESS_KEY = os.getenv("APP_ACCESS_KEY", "").strip()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

@st.dialog("🔐 Login Required")
def login_dialog():
    st.write("Please enter the access key to continue.")
    access_key = st.text_input("Access Key", type="password")
    if st.button("Login", type="primary", use_container_width=True):
        if access_key == APP_ACCESS_KEY:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Invalid Access Key")

if APP_ACCESS_KEY and not st.session_state.authenticated:
    login_dialog()
    st.stop()


# Defaults
DEFAULT_TOP_K = 50
DEFAULT_CHUNK_CANDIDATES = 70
INITIAL_DISPLAY_COUNT = 3
PAGINATION_STEP = 5

# Search input
query = st.text_input(
    "Search",
    placeholder="e.g., perjanjian kredit, laporan keuangan, perpajakan...",
    label_visibility="collapsed"
)

# Initialize display count for pagination
if "display_count" not in st.session_state:
    st.session_state.display_count = INITIAL_DISPLAY_COUNT

# Reset pagination on new search
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if query != st.session_state.last_query:
    st.session_state.display_count = INITIAL_DISPLAY_COUNT
    st.session_state.last_query = query

if st.button("Search", type="primary", use_container_width=True) or query:
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching documents..."):
            try:
                logger.info(f"Searching for query: {query}")
                response = requests.get(
                    SEARCH_API_URL,
                    params={
                        "query": query,
                        "top_k": DEFAULT_TOP_K,
                        "chunk_candidates": DEFAULT_CHUNK_CANDIDATES,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])

                if not results:
                    st.info("No results found. Try different keywords.")
                    logger.info("No results found.")
                else:
                    st.success(f"Found {len(results)} results")
                    logger.info(f"Found {len(results)} results.")
                    
                    # Show up to display_count results
                    current_display = min(st.session_state.display_count, len(results))
                    for idx, result in enumerate(results[:current_display], 1):
                        with st.container():
                            file_name = result.get("fileName", "Untitled")
                            ext = file_name.lower().split('.')[-1] if '.' in file_name else ''
                            if ext in ["zip", "rar", "7z"]:
                                badge = "📦 Archive"
                            elif ext in ["xlsx", "xls", "csv"]:
                                badge = "📊 Spreadsheet"
                            elif ext in ["jpg", "jpeg", "png", "gif"]:
                                badge = "🖼️ Image"
                            elif ext in ["doc", "docx"]:
                                badge = "📝 Document"
                            elif ext == "pdf":
                                badge = "📄 PDF"
                            else:
                                badge = "📄 File"

                            st.markdown(f"### {idx}. {file_name} `{badge}`")

                            web_url = result.get("webUrl")
                            if web_url:
                                col_btn1, col_btn2 = st.columns(2)
                                with col_btn1:
                                    st.link_button("📃 Open File", web_url, use_container_width=True)
                                with col_btn2:
                                    folder_url = (
                                        web_url.rsplit('/' + file_name, 1)[0]
                                        if file_name and file_name in web_url
                                        else web_url.rsplit('/', 1)[0]
                                    )
                                    st.link_button("📂 Open Folder", folder_url, use_container_width=True)

                            summary = result.get("summary", "No summary available")
                            with st.expander("📄 Summary", expanded=False):
                                st.markdown(summary)

                            st.divider()

                    # If more results exist, show See more button
                    if st.session_state.display_count < len(results):
                        if st.button("See more", use_container_width=True):
                            st.session_state.display_count = min(
                                st.session_state.display_count + PAGINATION_STEP, len(results)
                            )
                            st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to search service: {e}")
                st.info(f"Make sure the search service is running at {SEARCH_API_URL}")
                logger.error(f"Connection error: {e}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.error(f"Unexpected error: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <small>Document Search Stack v2.0 | Powered by Qdrant + Google Gemini</small>
    </div>
    """,
    unsafe_allow_html=True,
)
