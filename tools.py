import ast
import json
import re
from collections import Counter

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.tools import tool


@tool
def generate_chart(chart_type: str, data_json: str, title: str, x_key: str = "", y_key: str = "", z_key: str = "") -> str:
    """
    Generates an analytical chart.
    Args:
        chart_type: One of "bar", "line", "scatter", "pie", "donut", "treemap", "heatmap", "box", "histogram", "area", "funnel".
        data_json: A JSON string of the raw data rows to plot (e.g., '[{"category": "A", "value": 10}, ...]').
        title: The title of the chart.
        x_key: The dictionary key for the X axis (or names/categories for pie/treemap).
        y_key: The dictionary key for the Y axis (or values for pie/treemap).
        z_key: Optional. The dictionary key for the Z axis (values for heatmap, size for scatter bubble).
    Returns:
        A success string. The UI intercepts this to render natively.
    """
    try:
        import json
        json.loads(data_json)
    except Exception:
        return "Error: data_json must be a valid JSON string of an array of objects."
    return "Chart configuration successfully saved. The UI will display the chart."



def _generate_compact_eda_profile(data_string: str) -> str:
    if not data_string or len(data_string) < 10:
        return ""
    try:
        import pandas as pd
        import json
        import ast
        
        parsed = None
        try:
            parsed = json.loads(data_string)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(data_string)
            except Exception:
                pass
                
        if not parsed or not isinstance(parsed, list) or len(parsed) == 0:
            return ""
            
        # SQL tuples might be list of tuples without column names. Let's see if we can still run EDA.
        # It's better if they have names, but pandas handles list of tuples.
        df = pd.DataFrame(parsed)
        if df.empty or len(df) < 5:
            return ""
            
        warnings = []
        # Missing values
        for col in df.columns:
            missing_pct = df[col].isnull().mean()
            if missing_pct > 0.05:
                warnings.append(f"'{col}' has {int(missing_pct * 100)}% missing values")
                
        # Outliers using IQR
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                pct_outlier = len(outliers) / len(df)
                if pct_outlier > 0.01:
                    warnings.append(f"'{col}' has {len(outliers)} outliers")
                    
        if warnings:
            return "\n\n[EDA Profile: " + " | ".join(warnings) + "]"
        return ""
    except Exception:
        return ""


def build_sql_tools(database, llm):
    toolkit = SQLDatabaseToolkit(db=database, llm=llm)
    tools = toolkit.get_tools()
    
    for t in tools:
        if t.name == "sql_db_query":
            original_run = t._run
            def _wrapped_run(*args, **kwargs):
                res = original_run(*args, **kwargs)
                eda = _generate_compact_eda_profile(res)
                return str(res) + eda
            t._run = _wrapped_run
            
    tools = [tool for tool in tools if tool.name != "sql_db_query_checker"]
    tools.append(generate_chart)
    return tools


def _serialize_documents(documents):
    return json.dumps(documents, indent=2, default=str)


def _safe_limit(value, minimum=1, maximum=50):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = minimum
    return max(minimum, min(parsed, maximum))


def _load_query_json(query_json):
    if not query_json or not str(query_json).strip():
        return {}

    query = _parse_loose_json_like_value(query_json)

    if not isinstance(query, dict):
        raise ValueError("MongoDB query must decode to a JSON object.")

    return query


def _strip_code_fences(text):
    cleaned_text = str(text).strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.split("\n", 1)[1] if "\n" in cleaned_text else ""
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
    return cleaned_text.strip()


def _extract_balanced_segment(text, opening, closing):
    start_index = text.find(opening)
    if start_index == -1:
        return text.strip()

    depth = 0
    in_string = None
    escape_next = False

    for index in range(start_index, len(text)):
        character = text[index]

        if escape_next:
            escape_next = False
            continue

        if character == "\\" and in_string:
            escape_next = True
            continue

        if character in {'"', "'"}:
            if in_string == character:
                in_string = None
            elif in_string is None:
                in_string = character
            continue

        if in_string:
            continue

        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[start_index:index + 1].strip()

    return text[start_index:].strip()


def _parse_loose_json_like_value(value_text):
    cleaned_text = _strip_code_fences(value_text)

    if not cleaned_text:
        return {}

    for candidate_text in (
        cleaned_text,
        _extract_balanced_segment(cleaned_text, "[", "]"),
        _extract_balanced_segment(cleaned_text, "{", "}"),
    ):
        candidate_text = candidate_text.strip()
        if not candidate_text:
            continue

        try:
            return json.loads(candidate_text)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(candidate_text)
        except (ValueError, SyntaxError):
            pass

    raise ValueError("Could not parse structured input as JSON or Python literal.")


def _value_type_name(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


URL_PATTERN = re.compile(r"https?://[^\s\]\)\}>\x27\"\x22]+")


def _collect_url_values(value, urls):
    if isinstance(value, str):
        urls.update(match.group(0) for match in URL_PATTERN.finditer(value))
        return

    if isinstance(value, dict):
        for nested_value in value.values():
            _collect_url_values(nested_value, urls)
        return

    if isinstance(value, list):
        for item in value:
            _collect_url_values(item, urls)


def _extract_urls_from_documents(documents):
    urls = set()
    for document in documents:
        _collect_url_values(document, urls)
    return sorted(urls)


def _extract_linked_fields_from_documents(documents):
    linked_values = []

    def walk(value, path):
        if isinstance(value, str):
            if URL_PATTERN.search(value):
                linked_values.append({"path": path, "value": value})
            return

        if isinstance(value, dict):
            for key, nested_value in value.items():
                walk(nested_value, f"{path}.{key}" if path else str(key))
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]" if path else f"[{index}]")

    for document in documents:
        walk(document, "")

    return linked_values


def _collect_schema_fields(documents):
    field_stats = {}

    def record(path, value):
        field_path = path or "_value"
        entry = field_stats.setdefault(
            field_path,
            {"path": field_path, "occurrences": 0, "types": Counter(), "examples": []},
        )
        entry["occurrences"] += 1
        entry["types"][_value_type_name(value)] += 1

        if len(entry["examples"]) < 3:
            example = value
            if isinstance(value, (dict, list)):
                example = json.dumps(value, default=str)
            if example not in entry["examples"]:
                entry["examples"].append(example)

    def walk(value, path):
        if isinstance(value, dict):
            if path:
                record(path, value)
            for key, nested_value in value.items():
                next_path = f"{path}.{key}" if path else str(key)
                walk(nested_value, next_path)
            return

        if isinstance(value, list):
            record(path, value)
            for item in value[:5]:
                walk(item, f"{path}[]" if path else "[]")
            return

        record(path, value)

    for document in documents:
        walk(document, "")

    schema_fields = []
    for entry in sorted(field_stats.values(), key=lambda item: item["path"]):
        schema_fields.append(
            {
                "path": entry["path"],
                "occurrences": entry["occurrences"],
                "types": dict(entry["types"]),
                "examples": entry["examples"],
            }
        )

    return schema_fields


def _get_collection(database_connection, collection_name):
    if not collection_name or not str(collection_name).strip():
        raise ValueError("collection_name is required.")
    return database_connection.database[str(collection_name)]


def _validate_aggregation_pipeline(pipeline):
    allowed_stages = {
        "$addFields",
        "$count",
        "$group",
        "$limit",
        "$match",
        "$project",
        "$replaceRoot",
        "$replaceWith",
        "$set",
        "$skip",
        "$sort",
        "$unwind",
    }

    if not isinstance(pipeline, list):
        raise ValueError("MongoDB pipeline must decode to a JSON array.")

    for stage in pipeline:
        if not isinstance(stage, dict) or len(stage) != 1:
            raise ValueError("Each aggregation stage must be a single-key JSON object.")
        stage_name = next(iter(stage))
        if stage_name not in allowed_stages:
            raise ValueError(f"Unsupported aggregation stage: {stage_name}")


def _load_pipeline_json(pipeline_json):
    if not pipeline_json or not str(pipeline_json).strip():
        return []

    pipeline = _parse_loose_json_like_value(pipeline_json)
    if not isinstance(pipeline, list):
        raise ValueError("MongoDB pipeline must decode to a JSON array.")
    return pipeline


def build_mongo_tools(database_connection):
    @tool("list_mongo_collections")
    def list_mongo_collections() -> str:
        """List the collection names in the connected MongoDB database."""
        collections = database_connection.list_collection_names()
        return json.dumps(collections, indent=2)

    @tool("describe_mongo_collection")
    def describe_mongo_collection(collection_name: str, sample_limit: int = 20) -> str:
        """Summarize a MongoDB collection using sample documents, indexes, and discovered field paths."""
        safe_limit = _safe_limit(sample_limit, 1, 100)
        collection = _get_collection(database_connection, collection_name)
        documents = list(collection.find({}).limit(safe_limit))

        try:
            document_count = collection.count_documents({})
        except Exception:
            document_count = None

        summary = {
            "collection_name": collection_name,
            "document_count": document_count,
            "sample_limit": safe_limit,
            "sampled_documents": len(documents),
            "indexes": collection.index_information(),
            "field_paths": _collect_schema_fields(documents),
        }
        return json.dumps(summary, indent=2, default=str)

    @tool("sample_mongo_documents")
    def sample_mongo_documents(collection_name: str, limit: int = 5) -> str:
        """Return a sample of documents from one MongoDB collection."""
        safe_limit = _safe_limit(limit, 1, 50)
        collection = _get_collection(database_connection, collection_name)
        documents = list(collection.find({}).limit(safe_limit))
        res = _serialize_documents(documents)
        return res + _generate_compact_eda_profile(res)

    @tool("find_mongo_documents")
    def find_mongo_documents(collection_name: str, query_json: str = "{}", limit: int = 5) -> str:
        """Find read-only documents in a MongoDB collection using a JSON query."""
        query = _load_query_json(query_json)
        safe_limit = _safe_limit(limit, 1, 50)
        documents = database_connection.find_documents(collection_name, query, safe_limit)
        res = _serialize_documents(documents)
        return res + _generate_compact_eda_profile(res)

    @tool("count_mongo_documents")
    def count_mongo_documents(collection_name: str, query_json: str = "{}") -> str:
        """Count documents in a MongoDB collection that match a JSON query."""
        query = _load_query_json(query_json)
        collection = _get_collection(database_connection, collection_name)
        count = collection.count_documents(query)
        return json.dumps(
            {
                "collection_name": collection_name,
                "query": query,
                "count": count,
            },
            indent=2,
            default=str,
        )

    @tool("distinct_mongo_values")
    def distinct_mongo_values(collection_name: str, field_path: str, query_json: str = "{}", limit: int = 50) -> str:
        """Return the distinct values for a field in one MongoDB collection."""
        if not field_path or not str(field_path).strip():
            raise ValueError("field_path is required.")

        query = _load_query_json(query_json)
        safe_limit = _safe_limit(limit, 1, 200)
        collection = _get_collection(database_connection, collection_name)
        values = collection.distinct(str(field_path), query)
        limited_values = values[:safe_limit]
        return json.dumps(
            {
                "collection_name": collection_name,
                "field_path": field_path,
                "query": query,
                "distinct_values": limited_values,
                "returned": len(limited_values),
                "total_distinct": len(values),
            },
            indent=2,
            default=str,
        )

    @tool("aggregate_mongo_documents")
    def aggregate_mongo_documents(collection_name: str, pipeline_json: str = "[]", limit: int = 50) -> str:
        """Run a read-only MongoDB aggregation pipeline with only safe stages."""
        pipeline = _load_pipeline_json(pipeline_json)

        _validate_aggregation_pipeline(pipeline)
        safe_limit = _safe_limit(limit, 1, 200)
        has_limit_stage = any("$limit" in stage for stage in pipeline)
        pipeline_to_run = pipeline if has_limit_stage else [*pipeline, {"$limit": safe_limit}]
        collection = _get_collection(database_connection, collection_name)
        documents = list(collection.aggregate(pipeline_to_run, allowDiskUse=False))
        res = _serialize_documents(documents)
        return res + _generate_compact_eda_profile(res)

    @tool("extract_mongo_urls")
    def extract_mongo_urls(collection_name: str, query_json: str = "{}", limit: int = 50) -> str:
        """Return exact URL strings found anywhere in matched MongoDB documents."""
        query = _load_query_json(query_json)
        safe_limit = _safe_limit(limit, 1, 200)
        documents = database_connection.find_documents(collection_name, query, safe_limit)
        return json.dumps(_extract_urls_from_documents(documents), indent=2)

    @tool("extract_mongo_link_fields")
    def extract_mongo_link_fields(collection_name: str, query_json: str = "{}", limit: int = 50) -> str:
        """Return every URL-like field together with its document path and exact value."""
        query = _load_query_json(query_json)
        safe_limit = _safe_limit(limit, 1, 200)
        documents = database_connection.find_documents(collection_name, query, safe_limit)
        return json.dumps(_extract_linked_fields_from_documents(documents), indent=2, default=str)

    return [
        list_mongo_collections,
        describe_mongo_collection,
        sample_mongo_documents,
        find_mongo_documents,
        count_mongo_documents,
        distinct_mongo_values,
        aggregate_mongo_documents,
        extract_mongo_urls,
        extract_mongo_link_fields,
        generate_chart,
    ]
