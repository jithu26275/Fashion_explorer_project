import streamlit as st
from fashionapp import (
    genai_text_advice, expand_fashion_keywords,
    fetch_asos_products, fetch_fallback_products, show_products, render_colors_in_text
)

def run():
    st.title("💬 Text-Based Fashion Stylist")
    st.write("Type in what kind of outfit or style advice you want, and our AI stylist will help!")


    user_input = st.text_area(
        "✍️ Describe your fashion need:", 
        placeholder="e.g., I need a stylish outfit for a summer evening party",
        key="text_input"
    )


    if st.button("Get Fashion Suggestions"):
        if user_input.strip():
            ai_advice = genai_text_advice(user_input)
            keywords = expand_fashion_keywords(ai_advice)

            # save in session_state
            st.session_state["ai_advice"] = ai_advice
            st.session_state["keywords"] = keywords
        else:
            st.warning("⚠️ Please enter a description before asking for suggestions.")

    if "ai_advice" in st.session_state:
        st.subheader("🤖 AI Stylist Advice")
        st.markdown(
            f"""
            <div style="
                background-color: rgba(0, 0, 0, 0.6);
                padding: 15px 20px;
                border-radius: 12px;
                border: 1px solid #FFD700;
                font-size: 18px;
                line-height: 1.6;
                color: #FFD700;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            ">
                {render_colors_in_text(st.session_state['ai_advice'])}
            </div>
            """,
            unsafe_allow_html=True
        )

    if "keywords" in st.session_state:
        st.subheader("🛍️ Suggested Products")
        st.markdown(f"**Extracted keywords:** {', '.join(st.session_state['keywords'])}")

        keyword = st.selectbox("Pick a keyword to search products:", st.session_state["keywords"])
        use_fallback = st.checkbox("Use fallback dataset instead", value=False)

        if st.button("Find Products"):
            if not use_fallback:
                products = fetch_asos_products(keyword)
                if products:
                    show_products(products)
                else:
                    st.warning("No products found. Showing fallback dataset.")
                    show_products(fetch_fallback_products())
            else:
                show_products(fetch_fallback_products())
