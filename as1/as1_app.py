import streamlit as st
import time

from assignment1 import read_vocabulary, process_data, autocomplete_word


@st.cache_resource
def load_model():
    vocab = read_vocabulary("as3_file/shakespeare-edit.txt")
    model = process_data(vocab)
    return model


def main():
    st.title("Autocomplete App (Shakespeare)")
    st.caption("Type a prefix. Suggestions update as you type.")

    model = load_model()

    query = st.text_input("Enter a prefix", value="th")

    t0 = time.perf_counter()
    suggestions = autocomplete_word(query, model)
    dt_ms = (time.perf_counter() - t0) * 1000

    st.caption(f"Query time: {dt_ms:.2f} ms")

    if suggestions:
        st.subheader("Suggestions (Top 10)")
        st.write(suggestions)
    else:
        st.info("No suggestions found.")


if __name__ == "__main__":
    main()