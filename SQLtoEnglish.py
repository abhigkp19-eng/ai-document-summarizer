import streamlit as st
from llm import english_to_sql, sql_to_english
from CreateDatabase import run_query, create_db

create_db()

st.title("SQL ↔ Natural Language AI")

mode = st.selectbox("Choose Mode", ["English → SQL", "SQL → English"])
user_input = st.text_area("Enter query")

if st.button("Convert"):
    if mode == "English → SQL":
        sql = english_to_sql(user_input)
        st.code(sql)

        if st.button("Run Query"):
            result = run_query(sql)
            st.write(result)

    else:
        explanation = sql_to_english(user_input)
        st.write(explanation)