import json
import re

import pandas as pd
import streamlit as st
from database import (
    DatabaseConfig,
    MongoDatabaseConfig,
    create_database_from_config,
    create_database_from_uri,
    create_mongo_database_from_config,
)
from sql_agent import build_compact_messages, create_mongo_agent, create_sql_agent, message_content_to_text


def _clean_display_text(value):
    if value is None:
        return ""
    return str(value)


def _parse_json_answer(answer):
    if not isinstance(answer, str):
        return None

    try:
        return json.loads(answer)
    except json.JSONDecodeError:
        return None


def _chunk_items(items, size=3):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _metric_value(value):
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return value


def _is_url_string(value):
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def _is_image_url(value):
    if not _is_url_string(value):
        return False

    lowered_value = value.lower().split("?", 1)[0].split("#", 1)[0]
    return lowered_value.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"))


MD_IMAGE_PATTERN = re.compile(r"!\[(?P<label>[^\]]*)\]\((?P<url>[^)]+)\)")
MD_LINK_PATTERN = re.compile(r"(?<!\!)\[(?P<label>[^\]]+)\]\((?P<url>[^)]+)\)")


def _extract_markdown_resources(value):
    if not isinstance(value, str):
        return []

    resources = []
    for match in MD_IMAGE_PATTERN.finditer(value):
        resources.append(
            {
                "kind": "image",
                "label": _clean_display_text(match.group("label")).strip() or "Open image",
                "url": match.group("url").strip().strip("\"'").strip(),
            }
        )

    for match in MD_LINK_PATTERN.finditer(value):
        resources.append(
            {
                "kind": "link",
                "label": _clean_display_text(match.group("label")).strip() or "Open link",
                "url": match.group("url").strip().strip("\"'").strip(),
            }
        )

    return resources


def _is_resource_value(value):
    return _is_url_string(value) or bool(_extract_markdown_resources(value))


def _clean_resource_text(value):
    if not isinstance(value, str):
        return _clean_display_text(value)

    cleaned_text = value
    cleaned_text = MD_IMAGE_PATTERN.sub(lambda match: _clean_display_text(match.group("label")).strip() or "Open image", cleaned_text)
    cleaned_text = MD_LINK_PATTERN.sub(lambda match: _clean_display_text(match.group("label")).strip() or "Open link", cleaned_text)
    return _clean_display_text(cleaned_text)


def _contains_url(value):
    if _is_resource_value(value):
        return True

    if isinstance(value, dict):
        return any(_contains_url(nested_value) for nested_value in value.values())

    if isinstance(value, list):
        return any(_contains_url(item) for item in value)

    return False


def _render_url_value(label, value):
    safe_label = _clean_display_text(label.replace("_", " ").title())

    st.link_button(safe_label, value)


def _render_scalar_value(label, value):
    safe_label = _clean_display_text(label.replace("_", " ").title())

    if _is_url_string(value):
        _render_url_value(label, value)
        return

    markdown_resources = _extract_markdown_resources(value)
    if markdown_resources:
        clean_text = _clean_resource_text(value)
        if clean_text and clean_text.strip() and clean_text.strip() not in {resource["label"] for resource in markdown_resources}:
            st.markdown(f"**{safe_label}**: {clean_text}")
        else:
            st.caption(safe_label)

        for resource in markdown_resources:
            st.link_button(resource["label"], resource["url"])
        return

    st.markdown(f"**{safe_label}**: {_clean_display_text(value)}")


def _render_list_value(label, value):
    safe_label = _clean_display_text(label.replace("_", " ").title())
    if not value:
        st.markdown(f"**{safe_label}**: []")
        return

    if all(_is_resource_value(item) for item in value):
        st.caption(safe_label)
        for index, item in enumerate(value):
            _render_scalar_value(f"{safe_label} {index + 1}", item)
        return

    if all(isinstance(item, dict) for item in value) and any(_contains_url(item) for item in value):
        st.caption(safe_label)
        for index, item in enumerate(value):
            _render_record_block(item, f"{safe_label} {index + 1}")
        return

    if all(isinstance(item, dict) for item in value):
        st.json(value)
        return

    st.markdown(f"**{safe_label}**: {', '.join(_clean_display_text(item) for item in value)}")


def _render_structure_value(label, value):
    safe_label = _clean_display_text(label.replace("_", " ").title())

    if _is_url_string(value):
        _render_url_value(label, value)
        return

    if isinstance(value, dict):
        if _contains_url(value):
            with st.expander(safe_label, expanded=False):
                for nested_key, nested_value in value.items():
                    _render_structure_value(nested_key, nested_value)
            return

        with st.expander(safe_label, expanded=False):
            st.json(value)
        return

    if isinstance(value, list):
        _render_list_value(label, value)
        return

    _render_scalar_value(label, value)


def _split_record_items(record):
    scalar_items = []
    nested_items = []

    for key, value in record.items():
        if isinstance(value, (list, dict)):
            nested_items.append((key, value))
        else:
            scalar_items.append((key, value))

    return scalar_items, nested_items


def _render_record_block(record, title):
    with st.expander(_clean_display_text(title), expanded=False):
        if not isinstance(record, dict):
            st.markdown(_clean_display_text(record))
            return

        scalar_items, nested_items = _split_record_items(record)
        metric_items = [item for item in scalar_items if not _is_resource_value(item[1])]
        link_items = [item for item in scalar_items if _is_resource_value(item[1])]

        if metric_items:
            _render_metrics(metric_items)

        if link_items:
            st.caption("Links")
            for label, value in link_items:
                _render_scalar_value(label, value)

        for key, value in nested_items:
            _render_structure_value(key, value)


def _render_metrics(items):
    if not items:
        return

    for batch in _chunk_items(items, 3):
        columns = st.columns(len(batch))
        for column, (label, value) in zip(columns, batch):
            with column:
                st.metric(_clean_display_text(label.replace("_", " ").title()), _metric_value(_clean_display_text(value)))


def _render_sql_payload(payload):
    st.caption("SQL result")

    if isinstance(payload, list):
        if not payload:
            st.info("No rows returned.")
            return

        if all(isinstance(item, dict) for item in payload):
            if any(_contains_url(item) for item in payload):
                st.caption(f"{len(payload)} row(s) with links")
                for index, row in enumerate(payload):
                    _render_record_block(row, f"Row {index + 1}")
            else:
                st.dataframe(payload, use_container_width=True, hide_index=True)
                with st.expander("Preview JSON", expanded=False):
                    st.json(payload[:5])
        else:
            st.markdown("\n".join(_clean_display_text(item) for item in payload))
        return

    if isinstance(payload, dict):
        table_like_key = None
        table_like_rows = None
        for candidate in ("rows", "data", "results", "records", "items"):
            candidate_value = payload.get(candidate)
            if isinstance(candidate_value, list):
                table_like_key = candidate
                table_like_rows = candidate_value
                break

        scalar_items = [
            (key, value)
            for key, value in payload.items()
            if key != table_like_key and not isinstance(value, (list, dict))
        ]
        metric_items = [item for item in scalar_items if not _is_resource_value(item[1])]
        link_items = [item for item in scalar_items if _is_resource_value(item[1])]

        if metric_items:
            _render_metrics(metric_items)

        if link_items:
            st.caption("Links")
            for label, value in link_items:
                _render_scalar_value(label, value)

        if table_like_rows is not None:
            if table_like_rows and all(isinstance(item, dict) for item in table_like_rows):
                if any(_contains_url(item) for item in table_like_rows):
                    st.caption(f"{len(table_like_rows)} row(s) with links")
                    for index, row in enumerate(table_like_rows):
                        _render_record_block(row, f"Row {index + 1}")
                else:
                    st.dataframe(table_like_rows, use_container_width=True, hide_index=True)
            else:
                st.markdown("\n".join(_clean_display_text(item) for item in table_like_rows))

        nested_items = [
            (key, value)
            for key, value in payload.items()
            if key != table_like_key and isinstance(value, (list, dict))
        ]
        for key, value in nested_items:
            _render_structure_value(key, value)
        return

    st.markdown(_clean_display_text(payload))


def _render_mongo_document(document, index):
    title_value = document.get("_id", f"Document {index + 1}") if isinstance(document, dict) else f"Document {index + 1}"
    if not isinstance(document, dict):
        st.markdown(_clean_display_text(document))
        return

    _render_record_block(document, f"Document {index + 1}: {title_value}")


def _render_mongo_payload(payload):
    st.caption("MongoDB result")

    if isinstance(payload, list):
        if not payload:
            st.info("No documents returned.")
            return

        if all(isinstance(item, dict) for item in payload):
            st.caption(f"{len(payload)} document(s)")
            for index, document in enumerate(payload):
                _render_mongo_document(document, index)
        else:
            st.markdown("\n".join(_clean_display_text(item) for item in payload))
        return

    if isinstance(payload, dict):
        if "count" in payload:
            count_value = payload.get("count")
            metric_candidates = [
                ("count", count_value),
                ("collection_name", payload.get("collection_name")),
                ("returned", payload.get("returned")),
            ]
            _render_metrics([(label, value) for label, value in metric_candidates if value is not None])

        if payload.get("query") is not None:
            with st.expander("Query", expanded=False):
                st.json(payload["query"])

        if "document_count" in payload or "sampled_documents" in payload:
            metric_candidates = [
                ("document_count", payload.get("document_count")),
                ("sampled_documents", payload.get("sampled_documents")),
                ("sample_limit", payload.get("sample_limit")),
            ]
            _render_metrics([(label, value) for label, value in metric_candidates if value is not None])

        if payload.get("field_paths"):
            with st.expander("Field paths", expanded=True):
                st.dataframe(payload["field_paths"], use_container_width=True, hide_index=True)

        if payload.get("distinct_values") is not None:
            with st.expander("Distinct values", expanded=True):
                distinct_values = payload["distinct_values"]
                if distinct_values and all(isinstance(item, dict) for item in distinct_values):
                    st.dataframe(distinct_values, use_container_width=True, hide_index=True)
                elif distinct_values:
                    st.dataframe({"value": distinct_values}, use_container_width=True, hide_index=True)
                else:
                    st.info("No distinct values returned.")

        if payload.get("indexes") is not None:
            with st.expander("Indexes", expanded=False):
                st.json(payload["indexes"])

        remaining_items = {
            key: value
            for key, value in payload.items()
            if key not in {"count", "collection_name", "returned", "document_count", "sampled_documents", "sample_limit", "field_paths", "distinct_values", "indexes", "query", "total_distinct"}
        }
        for key, value in remaining_items.items():
            if isinstance(value, (list, dict)):
                _render_structure_value(key, value)
            else:
                _render_scalar_value(key, value)
        return

    st.markdown(_clean_display_text(payload))


def render_answer(answer):
    payload = _parse_json_answer(answer)

    if st.session_state.connection_kind == "mongodb":
        if payload is None:
            if _contains_url(answer):
                _render_scalar_value("Answer", answer)
            else:
                st.markdown(_clean_display_text(answer))
            return
        _render_mongo_payload(payload)
        return

    if payload is None:
        if _contains_url(answer):
            _render_scalar_value("Answer", answer)
        else:
            st.markdown(_clean_display_text(answer))
        return

    _render_sql_payload(payload)


st.set_page_config(
    page_title="SQL Agent",
    page_icon="🗃️",
    layout="wide",
)


CUSTOM_CSS = """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(56, 189, 248, 0.18), transparent 32%),
            radial-gradient(circle at top right, rgba(34, 197, 94, 0.14), transparent 28%),
            linear-gradient(180deg, #050816 0%, #0a1020 42%, #050816 100%);
        color: #e5eefb;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(7, 11, 20, 0.96), rgba(10, 16, 32, 0.96));
        border-right: 1px solid rgba(148, 163, 184, 0.12);
    }

    .hero-card, .panel-card {
        background: rgba(8, 13, 28, 0.76);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(18px);
        border-radius: 22px;
        padding: 1.5rem 1.6rem;
    }

    .hero-card h1 {
        font-size: 2.35rem;
        line-height: 1.05;
        margin: 0;
        color: #f8fbff;
    }

    .hero-card p,
    .panel-card p,
    .panel-card li,
    .hero-card li {
        color: rgba(226, 232, 240, 0.82);
        font-size: 0.98rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.24);
        color: #8bd8ff;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        width: fit-content;
    }

    .connection-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0.7rem;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.24);
        color: #9af0be;
        font-size: 0.85rem;
    }

    .stButton button {
        background: linear-gradient(135deg, #22c55e 0%, #38bdf8 100%);
        color: #04111f;
        border: none;
        border-radius: 14px;
        font-weight: 700;
        padding: 0.65rem 1rem;
    }

    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextArea textarea {
        background: rgba(15, 23, 42, 0.95) !important;
        color: #eff6ff !important;
        border: 1px solid rgba(148, 163, 184, 0.24) !important;
        border-radius: 14px !important;
    }

    [data-testid="stChatMessage"] {
        background: rgba(8, 13, 28, 0.62);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 0.25rem 0.2rem;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.12);
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li {
        color: #f8fbff !important;
        font-size: 1.05rem;
        line-height: 1.6;
    }
    
    [data-testid="stChatMessage"] code {
        background: rgba(56, 189, 248, 0.1) !important;
        color: #8bd8ff !important;
        padding: 0.2rem 0.4rem;
        border-radius: 6px;
    }

</style>
"""


DEFAULT_MODEL_NAME = "qwen/qwen3-32b"


def initialize_session_state():
    defaults = {
        "chat_history": [],
        "agent": None,
        "database": None,
        "connection_label": None,
        "groq_api_key": "",
        "model_name": DEFAULT_MODEL_NAME,
        "connection_kind": "sql",
        "connection_type": "postgresql",
        "custom_uri": "",
        "db_host": "localhost",
        "db_port": "5432",
        "db_name": "",
        "db_username": "",
        "db_password": "",
        "odbc_driver": "ODBC Driver 18 for SQL Server",
        "mongodb_uri": "mongodb://localhost:27017",
        "mongodb_database": "",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <div class="badge">Secure database chat</div>
            <h1>Connect to SQL or MongoDB and ask questions in plain English.</h1>
            <p>
                Use the sidebar to enter your Groq API key and database credentials, then chat with
                your database using a dark, focused interface.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_database_from_sidebar():
    if st.session_state.connection_kind == "mongodb":
        config = MongoDatabaseConfig(
            uri=st.session_state.mongodb_uri.strip(),
            database=st.session_state.mongodb_database.strip(),
        )
        return create_mongo_database_from_config(config)

    custom_uri = st.session_state.custom_uri.strip()
    if custom_uri:
        return create_database_from_uri(custom_uri)

    config = DatabaseConfig(
        dialect=st.session_state.connection_type,
        host=st.session_state.db_host.strip(),
        port=int(st.session_state.db_port) if st.session_state.db_port else None,
        database=st.session_state.db_name.strip(),
        username=st.session_state.db_username.strip(),
        password=st.session_state.db_password,
        odbc_driver=st.session_state.odbc_driver.strip(),
    )
    return create_database_from_config(config)


def connect_database():
    if not st.session_state.groq_api_key.strip():
        st.error("Enter your Groq API key first.")
        return

    try:
        database = build_database_from_sidebar()
        agent = create_sql_agent(
            database,
            st.session_state.groq_api_key.strip(),
            model_name=st.session_state.model_name.strip() or DEFAULT_MODEL_NAME,
        ) if st.session_state.connection_kind == "sql" else create_mongo_agent(
            database,
            st.session_state.groq_api_key.strip(),
            model_name=st.session_state.model_name.strip() or DEFAULT_MODEL_NAME,
        )
        st.session_state.database = database
        st.session_state.agent = agent
        st.session_state.chat_history = []

        if st.session_state.connection_kind == "mongodb":
            st.session_state.connection_label = f"MongoDB @ {st.session_state.mongodb_database}"
        elif st.session_state.custom_uri.strip():
            st.session_state.connection_label = database.dialect
        else:
            st.session_state.connection_label = (
                f"{st.session_state.connection_type.title()} @ "
                f"{st.session_state.db_host}:{st.session_state.db_port or 'default'}"
            )

        st.success("Database connected successfully.")
    except Exception as exc:
        st.session_state.agent = None
        st.session_state.database = None
        st.session_state.connection_label = None
        st.error(f"Connection failed: {exc}")


def render_sidebar():
    with st.sidebar:
        st.markdown("## Connection")
        st.session_state.groq_api_key = st.text_input(
            "Groq API key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
        )
        st.session_state.model_name = DEFAULT_MODEL_NAME
        st.caption(f"Model: {DEFAULT_MODEL_NAME}")

        st.session_state.connection_kind = st.selectbox(
            "Connection type",
            ["sql", "mongodb"],
            format_func=lambda value: {
                "sql": "SQL database",
                "mongodb": "MongoDB",
            }[value],
        )

        if st.session_state.connection_kind == "mongodb":
            st.session_state.mongodb_uri = st.text_input(
                "MongoDB URI",
                value=st.session_state.mongodb_uri,
                placeholder="mongodb://localhost:27017",
            )
            st.session_state.mongodb_database = st.text_input(
                "MongoDB database",
                value=st.session_state.mongodb_database,
                placeholder="database_name",
            )
        else:
            st.session_state.connection_type = st.selectbox(
                "Database type",
                ["postgresql", "mysql", "mssql"],
                format_func=lambda value: {
                    "postgresql": "PostgreSQL",
                    "mysql": "MySQL",
                    "mssql": "SQL Server",
                }[value],
            )

            st.session_state.custom_uri = st.text_input(
                "Custom SQLAlchemy URI",
                value=st.session_state.custom_uri,
                placeholder="Optional: paste a full SQLAlchemy connection string",
            )

            st.caption("If you fill the custom URI, the fields below are ignored.")
            st.session_state.db_host = st.text_input("Host", value=st.session_state.db_host)
            st.session_state.db_port = st.text_input("Port", value=st.session_state.db_port)
            st.session_state.db_name = st.text_input("Database name", value=st.session_state.db_name)
            st.session_state.db_username = st.text_input("Username", value=st.session_state.db_username)
            st.session_state.db_password = st.text_input(
                "Password",
                value=st.session_state.db_password,
                type="password",
            )
            st.caption("Leave the password blank for SQL Server Windows integrated auth.")
            st.session_state.odbc_driver = st.text_input(
                "SQL Server ODBC driver",
                value=st.session_state.odbc_driver,
            )

        st.button("Connect", use_container_width=True, on_click=connect_database)

        if st.session_state.connection_label:
            st.markdown(
                f"<div class='connection-pill'>Connected: {st.session_state.connection_label}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            """
            <div class="panel-card">
                <p><strong>Notes</strong></p>
                <ul>
                    <li>MongoDB uses <code>pymongo</code> and read-only collection tools.</li>
                    <li>PostgreSQL uses <code>psycopg</code>.</li>
                    <li>MySQL uses <code>pymysql</code>.</li>
                    <li>SQL Server uses <code>pyodbc</code> and an installed ODBC driver.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_chat():
    if not st.session_state.agent:
        st.markdown(
            """
            <div class="panel-card">
                <p>Connect a database from the sidebar to begin chatting.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["content"])
                if message.get("queries"):
                    with st.expander("Executed Query", expanded=False):
                        for q in message["queries"]:
                            st.code(q, language="sql" if st.session_state.connection_kind == "sql" else "json")
                if message.get("charts"):
                    for chart_args in message["charts"]:
                        try:
                            chart_type = chart_args.get("chart_type")
                            data_json = chart_args.get("data_json", "[]")
                            title = chart_args.get("title", "Chart")
                            x_key = chart_args.get("x_key")
                            y_key = chart_args.get("y_key")
                            z_key = chart_args.get("z_key")
                            
                            import json
                            import pandas as pd
                            df = pd.DataFrame(json.loads(data_json))
                            if df.empty:
                                st.warning("The AI called generate_chart but provided no data. It likely hallucinated the tool call or failed to query the database first.")
                                continue
                            
                            st.markdown(f"### {title}")
                            
                            if chart_type in ("heatmap", "box", "histogram"):
                                import matplotlib.pyplot as plt
                                import seaborn as sns
                                fig, ax = plt.subplots(figsize=(8, 6))
                                fig.patch.set_alpha(0) # transparent background
                                ax.set_facecolor('none')
                                if chart_type == "heatmap":
                                    if z_key and x_key and y_key:
                                        pivoted = df.pivot(index=y_key, columns=x_key, values=z_key)
                                        sns.heatmap(pivoted, ax=ax, cmap="Blues", annot=True)
                                    else:
                                        numeric_df = df.select_dtypes(include='number')
                                        sns.heatmap(numeric_df.corr(), ax=ax, cmap="Blues", annot=True)
                                elif chart_type == "box":
                                    sns.boxplot(data=df, x=x_key, y=y_key, ax=ax)
                                elif chart_type == "histogram":
                                    sns.histplot(data=df, x=x_key, ax=ax)
                                
                                # Make labels white for dark theme
                                ax.tick_params(colors='white')
                                ax.xaxis.label.set_color('white')
                                ax.yaxis.label.set_color('white')
                                st.pyplot(fig)
                            else:
                                import plotly.express as px
                                if chart_type == "bar":
                                    fig = px.bar(df, x=x_key, y=y_key)
                                elif chart_type == "line":
                                    fig = px.line(df, x=x_key, y=y_key)
                                elif chart_type == "scatter":
                                    fig = px.scatter(df, x=x_key, y=y_key, size=z_key if z_key else None)
                                elif chart_type == "pie":
                                    fig = px.pie(df, names=x_key, values=y_key)
                                elif chart_type == "donut":
                                    fig = px.pie(df, names=x_key, values=y_key, hole=0.5)
                                elif chart_type == "area":
                                    fig = px.area(df, x=x_key, y=y_key)
                                elif chart_type == "funnel":
                                    fig = px.funnel(df, x=y_key, y=x_key)
                                elif chart_type == "treemap":
                                    fig = px.treemap(df, path=[x_key], values=y_key)
                                else:
                                    fig = px.bar(df, x=x_key, y=y_key)
                                
                                fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="white")
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render chart: {e}")

    prompt = st.chat_input("Ask about your database")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            queries = []
            charts = []
            try:
                messages = build_compact_messages(st.session_state.chat_history, prompt)
                response = st.session_state.agent.invoke({"messages": messages})
                answer = message_content_to_text(response["messages"][-1].content)
                
                for msg in response.get("messages", []):
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            name = tc.get("name")
                            args = tc.get("args", {})
                            if name == "sql_db_query" and "query" in args:
                                queries.append(args["query"])
                            elif name in ("find_mongo_documents", "count_mongo_documents", "extract_mongo_urls", "extract_mongo_link_fields", "distinct_mongo_values") and "query_json" in args:
                                queries.append(args["query_json"])
                            elif name == "aggregate_mongo_documents" and "pipeline_json" in args:
                                queries.append(args["pipeline_json"])
                            elif name == "generate_chart":
                                charts.append(args)
            except Exception as exc:
                answer = f"I ran into an error: {exc}"

            if not answer and charts:
                answer = "Here is the visualization you requested:"
            elif not answer and not queries and not charts:
                answer = "I'm sorry, I couldn't generate a response."
                
            render_answer(answer)
            if queries:
                with st.expander("Executed Query", expanded=False):
                    for q in queries:
                        st.code(q, language="sql" if st.session_state.connection_kind == "sql" else "json")
            if charts:
                for chart_args in charts:
                        try:
                            chart_type = chart_args.get("chart_type")
                            data_json = chart_args.get("data_json", "[]")
                            title = chart_args.get("title", "Chart")
                            x_key = chart_args.get("x_key")
                            y_key = chart_args.get("y_key")
                            z_key = chart_args.get("z_key")
                            
                            import json
                            import pandas as pd
                            df = pd.DataFrame(json.loads(data_json))
                            if df.empty:
                                st.warning("The AI called generate_chart but provided no data. It likely hallucinated the tool call or failed to query the database first.")
                                continue
                            
                            st.markdown(f"### {title}")
                            
                            if chart_type in ("heatmap", "box", "histogram"):
                                import matplotlib.pyplot as plt
                                import seaborn as sns
                                fig, ax = plt.subplots(figsize=(8, 6))
                                fig.patch.set_alpha(0) # transparent background
                                ax.set_facecolor('none')
                                if chart_type == "heatmap":
                                    if z_key and x_key and y_key:
                                        pivoted = df.pivot(index=y_key, columns=x_key, values=z_key)
                                        sns.heatmap(pivoted, ax=ax, cmap="Blues", annot=True)
                                    else:
                                        numeric_df = df.select_dtypes(include='number')
                                        sns.heatmap(numeric_df.corr(), ax=ax, cmap="Blues", annot=True)
                                elif chart_type == "box":
                                    sns.boxplot(data=df, x=x_key, y=y_key, ax=ax)
                                elif chart_type == "histogram":
                                    sns.histplot(data=df, x=x_key, ax=ax)
                                
                                # Make labels white for dark theme
                                ax.tick_params(colors='white')
                                ax.xaxis.label.set_color('white')
                                ax.yaxis.label.set_color('white')
                                st.pyplot(fig)
                            else:
                                import plotly.express as px
                                if chart_type == "bar":
                                    fig = px.bar(df, x=x_key, y=y_key)
                                elif chart_type == "line":
                                    fig = px.line(df, x=x_key, y=y_key)
                                elif chart_type == "scatter":
                                    fig = px.scatter(df, x=x_key, y=y_key, size=z_key if z_key else None)
                                elif chart_type == "pie":
                                    fig = px.pie(df, names=x_key, values=y_key)
                                elif chart_type == "donut":
                                    fig = px.pie(df, names=x_key, values=y_key, hole=0.5)
                                elif chart_type == "area":
                                    fig = px.area(df, x=x_key, y=y_key)
                                elif chart_type == "funnel":
                                    fig = px.funnel(df, x=y_key, y=x_key)
                                elif chart_type == "treemap":
                                    fig = px.treemap(df, path=[x_key], values=y_key)
                                else:
                                    fig = px.bar(df, x=x_key, y=y_key)
                                
                                fig.update_layout(
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="white")
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Failed to render chart: {e}")

    st.session_state.chat_history.append({"role": "assistant", "content": answer, "queries": queries, "charts": charts})


def main():
    initialize_session_state()
    render_header()
    render_sidebar()
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    render_chat()


main()
