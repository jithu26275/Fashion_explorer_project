import streamlit as st
import random

# --- SETTINGS ---
# icons = ["👗", "👜", "👠", "👒", "👕", "👖"]
icon = ["LOGO.jpg","LOGO2.jpg","LOGO3.jpg","LOGO4.jpg"]
page_icon = random.choice(icon)
st.set_page_config(page_title="Fashion App", page_icon=page_icon, layout="wide")

background_images = [
    "https://images.pexels.com/photos/794064/pexels-photo-794064.jpeg",
    "https://images.pexels.com/photos/2983464/pexels-photo-2983464.jpeg",
    "https://images.pexels.com/photos/298863/pexels-photo-298863.jpeg",
    "https://cdn.pixabay.com/photo/2022/07/06/12/58/woman-7305088_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/21/12/40/woman-1845148_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/29/09/25/blonde-1868701_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/03/26/22/13/man-1281562_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/08/01/09/41/people-2564027_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/03/20/15/13/wrist-watch-2159351_1280.jpg",
    "https://cdn.pixabay.com/photo/2019/10/20/14/58/portrait-4563909_1280.jpg",
    "https://cdn.pixabay.com/photo/2021/03/03/10/35/man-6065000_1280.jpg",
    "https://cdn.pixabay.com/photo/2016/11/19/15/58/camera-1840054_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/07/09/00/29/woman-837156_1280.jpg",
    "https://cdn.pixabay.com/photo/2020/05/15/10/18/model-5173119_1280.jpg",
    "https://cdn.pixabay.com/photo/2021/06/14/02/32/man-6334818_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/01/18/17/14/girl-1990347_1280.jpg",
    "https://cdn.pixabay.com/photo/2017/03/20/17/44/wrist-watch-2159785_1280.jpg",
    "https://cdn.pixabay.com/photo/2021/07/21/04/35/woman-6482214_1280.jpg",
    "https://cdn.pixabay.com/photo/2015/11/06/16/23/fashion-1029438_1280.jpg",
    "https://cdn.pixabay.com/photo/2019/03/05/05/45/man-4035612_1280.jpg",
    "https://cdn.pixabay.com/photo/2021/12/07/02/38/woman-6851973_1280.jpg",
    "https://cdn.pixabay.com/photo/2021/10/08/11/56/man-6691157_1280.jpg",
    "https://www.pexels.com/photo/woman-in-pink-and-white-stripe-long-sleeve-shirt-wearing-black-hat-9136195.jpg"
]


bg_image = random.choice(background_images)

# --- CUSTOM CSS ---
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }}

    /* Sidebar background */
    section[data-testid="stSidebar"] {{
        background-color: rgba(0,0,0,0.7); /* semi-transparent black */
    }}

    /* Sidebar text */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{
        color: #ff69b4 !important; /* hot pink */
        font-weight: 600;
    }}

    /* Radio/selected option styles */
    div[data-baseweb="radio"] label span {{
        color: #ff69b4 !important; /* default option color */
        font-weight: 500;
    }}

    /* Highlight for selected radio option */
    div[data-baseweb="radio"] label[data-selected="true"] span {{
        color: #FFD700 !important; /* gold for selected */
        font-weight: 700;
    }}
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: -1;
    }}
    h1 {{
        font-size: 70px;
        color: #ff69b4;
        text-align: center;
        text-shadow: 2px 2px 4px #000000;
        animation: slideDown 1s ease-out;
    }}
    /* Fade-in animation for paragraphs */
    .fade-in {{
    font-size: 22px;
    color: #ffffff; /* pure white */
    font-weight: 400;
    text-align: center;
    animation: fadeIn 2s ease-in;

    background: rgba(0, 0, 0, 0.5); /* translucent black box */
    padding: 12px 18px;
    border-radius: 12px;
    display: inline-block;
    margin: 10px auto;
    }}
    p, .stMarkdown {{
        font-size: 22px;
        color: #f8f8f8;
        font-weight: 400;
        text-align: center;
        animation: fadeIn 2s ease-in;
    }}
    @keyframes slideDown {{
        0% {{ transform: translateY(-100px); opacity: 0; }}
        100% {{ transform: translateY(0); opacity: 1; }}
    }}
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    </style>
    <div class="overlay"></div>
    """,
    unsafe_allow_html=True
)

# logo = ["logo1.png","logo2.png"]
# logo_random = random.choice(logo)
logo = ["LOGO.jpg","LOGO2.jpg","LOGO3.jpg","LOGO4.jpg"]
logo_random = random.choice(logo)
# --- SIDEBAR NAVIGATION ---
st.sidebar.image(logo_random, width=120)

page = st.sidebar.radio(
    "Navigation",   # ✅ required label (for accessibility)
    ["Home", "Image", "Text", "Live"],
    label_visibility="collapsed"  # ✅ hides the label visually
)

# --- PAGE LOGIC ---
if page == "Home":
    st.title("🏠 Welcome to My Fashion App")
    st.markdown(
        """
        <p class="fade-in">
            This app is the prefect place for you to get suggestions from pictures you upload or also from prompts you type.
        </p>
        <p class="fade-in">
            My app will automatically detect objects from the image or it will collect key words based on your prompt and provide fashion advices or suggestions.
        </p>
        <p class="fade-in">
            Use the side bar to navigate through pages.
        </p>
        """,
        unsafe_allow_html=True
    )
elif page == "Image":
    import image
    image.run()

elif page == "Text":
    import text
    text.run()

elif page == "Live":
    import webcam
    # webcam.run()
