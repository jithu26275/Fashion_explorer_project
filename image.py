import streamlit as st
from fashionapp import (
    detect_items_with_colors, fashion_logic, genai_style_advice,
    fetch_asos_products, fetch_fallback_products, show_products,
    render_colors_in_text, expand_fashion_keywords
)

def run():
    st.title("🖼️ Image Detection Page")
    st.write("Upload an image to run detections here.")

    # --- Upload Image ---
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Run detection only when a new image is uploaded
        if "uploaded_path" not in st.session_state or st.session_state["uploaded_path"] != uploaded_file.name:
            with open("temp.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state["uploaded_path"] = uploaded_file.name

            # --- Run Detection Once ---
            detected, detected_colors = detect_items_with_colors("temp.jpg")
            rules = fashion_logic(detected)
            ai_advice = genai_style_advice(detected, detected_colors)
            expanded_keywords = expand_fashion_keywords(ai_advice)

            # Save results in session_state
            st.session_state["detected"] = detected
            st.session_state["detected_colors"] = detected_colors
            st.session_state["rules"] = rules
            st.session_state["ai_advice"] = ai_advice
            st.session_state["keywords"] = expanded_keywords

        # --- Read from session_state ---
        detected = st.session_state["detected"]
        detected_colors = st.session_state["detected_colors"]
        rules = st.session_state["rules"]
        ai_advice = st.session_state["ai_advice"]
        expanded_keywords = st.session_state["keywords"]

        # --- Colors by Item ---
        if detected_colors:
            st.subheader("🎨 Colors by Item")
            cols = st.columns(len(detected_colors))
            for i, (item, color) in enumerate(detected_colors.items()):
                with cols[i % len(cols)]:
                    st.markdown(f"**{item.capitalize()}**")
                    st.markdown(
                        f"<div style='width:100%;height:60px;border-radius:8px;background:{color};'></div>",
                        unsafe_allow_html=True
                    )
                    st.write(color)

        # --- Fashion Suggestions ---
        st.subheader("✨ Fashion Suggestions")
        with st.expander("📌 View Suggestions"):
            for s in rules:
                st.markdown(
                    f"""
                    <div style="
                        background-color: rgba(255, 105, 180, 0.15);
                        padding: 12px 18px;
                        margin-bottom: 10px;
                        border-radius: 12px;
                        border: 1px solid #ff69b4;
                        color: white;
                        font-size: 18px;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.2);
                    ">
                    {s}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- AI Stylist Advice ---
        if ai_advice:
            st.subheader("🤖 AI Stylist Advice")
            with st.expander("💡 View AI Recommendations"):
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
                        {render_colors_in_text(ai_advice)}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # --- Suggested Products ---
        if expanded_keywords:
            st.subheader("🛍️ Suggested Products")
            st.markdown(f"**Expanded keywords:** {', '.join(expanded_keywords)}")

            # Keep last selected keyword
            if "selected_keyword" not in st.session_state:
                st.session_state["selected_keyword"] = expanded_keywords[0]

            keyword = st.selectbox(
                "Pick a keyword to search products:",
                expanded_keywords,
                index=expanded_keywords.index(st.session_state["selected_keyword"])
                if st.session_state["selected_keyword"] in expanded_keywords
                else 0,
                key="keyword_select"
            )

            use_fallback = st.checkbox("Use fallback dataset instead", value=False)

            if st.button("Find Products"):
                st.session_state["selected_keyword"] = keyword  # save selection
                if not use_fallback:
                    products = fetch_asos_products(keyword)
                    if products:
                        st.session_state["products"] = products
                    else:
                        st.warning("No products found. Showing fallback dataset.")
                        st.session_state["products"] = fetch_fallback_products()
                else:
                    st.session_state["products"] = fetch_fallback_products()

            # Show products if available
            if "products" in st.session_state:
                show_products(st.session_state["products"])
