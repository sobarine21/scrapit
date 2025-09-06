import streamlit as st
import psycopg2
import pandas as pd
import zipfile
import io

st.title("Supabase Database CSV Exporter")

# Supabase Credentials Inputs
host = st.text_input("Host (e.g., db.ljnzkgwbtqoxpztwupli.supabase.co)")
port = st.text_input("Port (usually 5432)", value="5432")
dbname = st.text_input("Database Name", value="postgres")
user = st.text_input("Username", value="postgres")
password = st.text_input("Password", type="password")

if st.button("Export All Tables as CSV ZIP"):
    try:
        # Connect to Supabase Postgres
        conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        st.success("Connected to Supabase database!")

        # Get list of tables in public schema
        query_tables = """
            SELECT tablename FROM pg_tables WHERE schemaname='public';
        """
        tables_df = pd.read_sql(query_tables, conn)
        tables = tables_df['tablename'].tolist()

        if not tables:
            st.warning("No tables found in public schema.")
        else:
            # Create in-memory ZIP
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                for table in tables:
                    st.info(f"Exporting table: {table}...")

                    # Query full table
                    df = pd.read_sql(f'SELECT * FROM public."{table}"', conn)

                    # Save to CSV in memory
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    zip_file.writestr(f"{table}.csv", csv_bytes)

            st.success(f"Exported {len(tables)} tables.")

            # Prepare ZIP for download
            zip_buffer.seek(0)
            st.download_button(
                label="Download CSV ZIP",
                data=zip_buffer,
                file_name="supabase_tables.zip",
                mime="application/zip"
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
