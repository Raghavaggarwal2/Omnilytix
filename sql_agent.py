
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from database import create_database_from_uri, create_mongo_database_from_config, MongoDatabaseConfig
from tools import build_mongo_tools, build_sql_tools

DEFAULT_MODEL_NAME = "qwen/qwen3-32b"

chat_history = []
MAX_MODEL_HISTORY_MESSAGES = 6
MAX_MODEL_MESSAGE_CHARS = 700
MAX_MODEL_HISTORY_CHARS = 2200


def _strip_inline_markdown(text):
    if not text:
        return text
    return text


def _markdown_table_to_text(lines):
    if len(lines) < 2:
        return lines

    header_line = lines[0]
    separator_line = lines[1]
    if "|" not in header_line or "|" not in separator_line:
        return lines

    def split_row(row):
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    separator_cells = split_row(separator_line)
    if not separator_cells or not all(set(cell) <= {"-", ":"} for cell in separator_cells):
        return lines

    headers = split_row(header_line)
    body_lines = []
    for row in lines[2:]:
        if "|" not in row:
            body_lines.append(row)
            continue

        values = split_row(row)
        pairs = []
        for index, header in enumerate(headers):
            value = values[index] if index < len(values) else ""
            pairs.append(f"{header}: {value}")
        body_lines.append(" | ".join(pairs))

    return body_lines


def _normalize_answer_text(answer_text):
    if not isinstance(answer_text, str):
        return str(answer_text)

    answer_text = _strip_inline_markdown(answer_text)

    lines = answer_text.splitlines()
    normalized_lines = []
    index = 0

    while index < len(lines):
        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if "|" in current_line and "|" in next_line and set(next_line.replace("|", "").strip()) <= {"-", ":"}:
            table_lines = [current_line, next_line]
            index += 2
            while index < len(lines) and "|" in lines[index]:
                table_lines.append(lines[index])
                index += 1
            normalized_lines.extend(_markdown_table_to_text(table_lines))
            continue

        normalized_lines.append(current_line)
        index += 1

    return "\n".join(normalized_lines)


def _shorten_text(text, limit=MAX_MODEL_MESSAGE_CHARS):
    normalized_text = _normalize_answer_text(text)
    if len(normalized_text) <= limit:
        return normalized_text

    head = max(200, limit // 2)
    tail = max(100, limit // 6)
    return f"{normalized_text[:head]}\n...[truncated]...\n{normalized_text[-tail:]}"


def build_compact_messages(history, question):
    compact_messages = []
    total_chars = len(question)

    for message in reversed(history or []):
        role = message.get("role") if isinstance(message, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if role not in {"user", "assistant"} or content is None:
            continue

        compact_content = _shorten_text(content)
        message_size = len(compact_content)
        if compact_messages and (
            len(compact_messages) >= MAX_MODEL_HISTORY_MESSAGES
            or total_chars + message_size > MAX_MODEL_HISTORY_CHARS
        ):
            break

        compact_messages.append({"role": role, "content": compact_content})
        total_chars += message_size

    compact_messages.reverse()
    compact_messages.append({"role": "user", "content": _shorten_text(question, limit=MAX_MODEL_MESSAGE_CHARS)})
    return compact_messages

SYSTEM_PROMPT_TEMPLATE = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

If you see an [EDA Profile: ...] block in your tool results, proactively mention the data quality warnings (like missing values or outliers) to the user in your response.

CRITICAL INSTRUCTION: ALWAYS respond to the user in a conversational, natural language format. Use markdown bullet points, bolding, and short paragraphs to present your findings. NEVER output raw JSON arrays or objects to the user unless they explicitly ask for JSON. Do not use markdown tables.

Use the previous conversation to understand follow-up questions like
"now filter that by 2024" or "show more details for the second one".

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

If the user asks to plot, chart, or visualize the data, first query the database to get the data, and then MUST use the `generate_chart` tool to create the visual representation using the exact numbers you retrieved.



To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""

MONGO_SYSTEM_PROMPT_TEMPLATE = """
You are an agent designed to interact with a MongoDB database.

Given an input question, first list the collections, then inspect the most
relevant documents before answering. Use only the read-only tools available to
you.

Use describe_mongo_collection when you need collection structure, field names,
indexes, or a schema-like overview. Use count_mongo_documents when you need a
record count. Use distinct_mongo_values when you need unique values for a
specific field. Use aggregate_mongo_documents for read-only grouped or
summarized queries that need more than simple matching.

When using aggregate_mongo_documents, provide pipeline_json as a JSON array of
objects such as [{"$sort": {"price": 1}}, {"$limit": 5}]. Do not wrap the
pipeline in markdown fences. If the question can be answered with count,
distinct, sample, or find, prefer that tool instead of aggregation.

If the user asks for image links, URLs, or other exact strings stored in the
documents, use the exact values returned by the tools and do not rewrite,
sanitize, or summarize them.

If the user asks for links or images, prefer the extract_mongo_link_fields tool
so you can return the exact field path and exact stored URL for every match.

When returning links, preserve the full URL exactly as it appears in the tool
output. Do not shorten it, remove query parameters, or turn it into a rewritten
summary.


If you see an [EDA Profile: ...] block in your tool results, proactively mention the data quality warnings (like missing values or outliers) to the user in your response.

CRITICAL INSTRUCTION: ALWAYS respond to the user in a conversational, natural language format. Use markdown bullet points, bolding, and short paragraphs to present your findings. NEVER output raw JSON arrays or objects to the user unless they explicitly ask for JSON. Do not use markdown tables.

Never make any write operations such as insert, update, delete, or drop.
If the user asks to plot, chart, or visualize the data, first query the database to get the data, and then MUST use the `generate_chart` tool to create the visual representation using the exact numbers you retrieved.


Keep your answers grounded in the data you retrieved and be careful to use the
exact collection names returned by the tools.
"""


def create_sql_agent(database, groq_api_key, model_name=DEFAULT_MODEL_NAME):
    if not groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY to create the agent.")

    model = ChatGroq(groq_api_key=groq_api_key, model=model_name)
    tools = build_sql_tools(database, model)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(dialect=database.dialect, top_k=5)
    return create_agent(model, tools, system_prompt=system_prompt)


def create_mongo_agent(database_connection, groq_api_key, model_name=DEFAULT_MODEL_NAME):
    if not groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY to create the agent.")

    model = ChatGroq(groq_api_key=groq_api_key, model=model_name)
    tools = build_mongo_tools(database_connection)
    return create_agent(model, tools, system_prompt=MONGO_SYSTEM_PROMPT_TEMPLATE)


def message_content_to_text(content):
    if isinstance(content, str):
        return _normalize_answer_text(content)

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                text_parts.append(str(block["text"]))
        return _normalize_answer_text("\n".join(text_parts))

    return _normalize_answer_text(str(content))


def ask_agent(agent, question, history=None):
    if history is None:
        history = []
    messages = build_compact_messages(history, question)
    response = agent.invoke(
        {"messages": messages},
    )
    answer = message_content_to_text(response["messages"][-1].content)

    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )

    return answer


def main():
    try:
        groq_api_key = input("Enter Groq API Key: ").strip()
        database_kind = input("Enter database kind (sql/mongo): ").strip().lower()

        if database_kind in {"mongo", "mongodb", "nosql"}:
            uri = input("Enter MongoDB URI: ").strip()
            db_name = input("Enter MongoDB Database Name: ").strip()
            if not uri or not db_name:
                raise RuntimeError("MongoDB URI and Database Name are required.")
            config = MongoDatabaseConfig(uri=uri, database=db_name)
            database = create_mongo_database_from_config(config)
            agent = create_mongo_agent(database, groq_api_key)
        else:
            db_url = input("Enter SQL Database URI: ").strip()
            if not db_url:
                raise RuntimeError("SQL Database URI is required.")
            database = create_database_from_uri(db_url)
            agent = create_sql_agent(database, groq_api_key)
    except Exception as exc:
        print(f"Failed to initialize database agent: {exc}")
        return

    print("SQL agent ready. Type 'exit' or 'quit' to stop.")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            answer = ask_agent(agent, question, chat_history)
        except Exception as exc:
            answer = f"Sorry, I ran into an error: {exc}"

        print(f"Ai: {answer}\n")


if __name__ == "__main__":
    main()
str