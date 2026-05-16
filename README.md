## Database setup

The app supports PostgreSQL, MySQL, SQL Server, and MongoDB. You can either paste a full SQLAlchemy URI in the Streamlit sidebar, fill in the SQL connection fields, or switch the sidebar to MongoDB and enter a MongoDB URI plus database name.

For local CLI usage, create a `.env` file in the project root and set your Groq key plus a connection string:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql+psycopg://user:password@host:5432/database_name
```

MongoDB example:

```env
GROQ_API_KEY=your_groq_api_key
MONGODB_URI=mongodb://user:password@host:27017
MONGODB_DATABASE=database_name
DATABASE_KIND=mongodb
```

Supported URI formats:

- PostgreSQL: `postgresql+psycopg://user:password@host:5432/database_name`
- MySQL: `mysql+pymysql://user:password@host:3306/database_name`
- SQL Server: `mssql+pyodbc://user:password@host:1433/database_name?driver=ODBC+Driver+18+for+SQL+Server`

MongoDB uses `MONGODB_URI` plus `MONGODB_DATABASE`.

SQL Server also requires the matching Microsoft ODBC driver to be installed on your machine.

## Run it

Streamlit UI:

```bash
streamlit run app.py
```

CLI mode:

```bash
python sql_agent.py
```

If you want to test the connection first, make sure the database user has permission to read the tables you want the agent to query.
