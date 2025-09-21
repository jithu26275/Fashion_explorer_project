import streamlit as st
import random
from ultralytics import YOLO
from transformers import pipeline
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
import cv2
from sklearn.cluster import KMeans
import numpy as np
import json
import re
import requests

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load once (consider using st.cache_resource in Streamlit)
model = YOLO(r"C:\Users\JITHIN\Downloads\best.pt")
class_map = {
    "top": ["shirt", "t-shirt", "crop top", "over-sized t-shirt"],
    "bottom": ["jeans", "pants", "shorts", "skirt", "trousers"],
    "fulldress": ["gown", "jumpsuit"],
    "footwear": ["sneakers", "heels", "sandals", "boots"],
    "accessories": ["necklace", "bracelet", "watch", "sunglasses"],
    "bag": ["handbag", "backpack", "clutch", "tote"],
    "outerwear": ["jacket", "coat", "hoodie", "blazer"],
    "hat": ["cap", "beanie", "fedora", "bucket hat"]}

def extract_dominant_color(image, num_colors=1):
    """Return hex string of dominant color using k-means on a resized crop."""
    # image expected as BGR (cv2.imread)
    if image is None or image.size == 0:
        return None
    # convert to RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape
    pixels = img_rgb.reshape((-1, 3))
    # small sample (kmeans is heavy on large images)
    # if too many pixels, sample random subset
    if pixels.shape[0] > 5000:
        idx = np.random.choice(pixels.shape[0], 5000, replace=False)
        pixels_sample = pixels[idx]
    else:
        pixels_sample = pixels
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10)
    kmeans.fit(pixels_sample)
    colors = kmeans.cluster_centers_.astype(int)
    # return first cluster as hex
    r, g, b = colors[0]
    return f'#{r:02x}{g:02x}{b:02x}'

def detect_items_with_colors(image_path, num_colors=1):
    """
    Runs YOLO once and returns:
      - detected: ordered unique list of class names found
      - detected_colors: dict class_name -> hex_color
    """
    results = model.predict(image_path, verbose=False)
    detected = []
    detected_colors = {}
    image = cv2.imread(image_path)
    if image is None:
        return [], {}

    h, w = image.shape[:2]

    for box in results[0].boxes:
        cls_name = results[0].names[int(box.cls)]
        # preserve order but keep unique
        if cls_name not in detected:
            detected.append(cls_name)

        # get xyxy and clamp
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))

        # crop; skip tiny boxes
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue
        crop = image[y1:y2, x1:x2]
        # resize crop to speed up clustering
        if max(crop.shape[:2]) > 200:
            crop = cv2.resize(crop, (200, int(crop.shape[0] * 200 / crop.shape[1])), interpolation=cv2.INTER_AREA)

        color = extract_dominant_color(crop, num_colors=num_colors)
        if color:
            detected_colors[cls_name] = color

    return detected, detected_colors
def detect_classes(image_path):
    results = model.predict(image_path, verbose=False)
    detected = list(set([results[0].names[int(box.cls)] for box in results[0].boxes]))
    return detected
def fashion_logic(detected):
    suggestions = []
    if 'top' in detected and 'bottom' in detected and not any(cls in detected for cls in ['fulldress','footwear','accessories','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['accessories'])}")
        suggestions.append(f"{random.choice(class_map['footwear'])}, {random.choice(class_map['accessories'])}, {random.choice(class_map['bag'])}, {random.choice(class_map['hat'])} match nicely")
    
    elif 'top' in detected and 'outerwear' in detected and not any(cls in detected for cls in ['bottom','fulldress','footwear','accessories','bag','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['bottom'])}")
        suggestions.append(f"{random.choice(class_map['footwear'])}, {random.choice(class_map['bag'])}, {random.choice(class_map['hat'])} match nicely")

    elif 'top' in detected and 'bag' in detected and not any(cls in detected for cls in ['bottom','fulldress','footwear','accessories','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['outerwear'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, match nicely")
    
    elif 'top' in detected and 'hat' in detected and not any(cls in detected for cls in ['bottom','fulldress','footwear','accessories','bag','outerwear']):
        suggestions.append(f"Pair with {', '.join(class_map['bottom'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, match nicely")

    elif 'top' in detected and 'accessories' in detected and not any(cls in detected for cls in ['bottom','fulldress','footwear','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['bottom'])}")
        suggestions.append(f"{random.choice(class_map['bag'])}, {random.choice(class_map['footwear'])}, {random.choice(class_map['hat'])} match nicely")

    elif 'fulldress' in detected and 'footwear' in detected and not any(cls in detected for cls in ['top','bottom','accessories','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['accessories'])}")
        suggestions.append(f"{random.choice(class_map['bag'])}, {random.choice(class_map['hat'])}, match nicely")

    elif 'fulldress' in detected and 'accessories' in detected and not any(cls in detected for cls in ['top','bottom','footwear','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['hat'])}")
        suggestions.append(f"{random.choice(class_map['bag'])}, {random.choice(class_map['footwear'])}, match nicely")

    elif 'fulldress' in detected and 'bag' in detected and not any(cls in detected for cls in ['top','bottom','footwear','accessories','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['footwear'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['hat'])}, match nicely")

    elif 'fulldress' in detected and 'hat' in detected and not any(cls in detected for cls in ['top','bottom','footwear','accessories','bag','outerwear']):
        suggestions.append(f"Pair with {', '.join(class_map['bag'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, match nicely")

    elif 'fulldress' in detected and 'bottom' in detected and not any(cls in detected for cls in ['top','footwear','accessories','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['bag'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, match nicely")

    elif 'bottom' in detected and 'footwear' in detected and not any(cls in detected for cls in ['top','fulldress','accessories','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['top'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['bag'])}, match nicely")

    elif 'bottom' in detected and 'bag' in detected and not any(cls in detected for cls in ['top','fulldress','footwear','accessories','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['top'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, match nicely")

    elif 'bottom' in detected and 'hat' in detected and not any(cls in detected for cls in ['top','fulldress','footwear','accessories','bag','outerwear']):
        suggestions.append(f"Pair with {', '.join(class_map['top'])}")
        suggestions.append(f"{random.choice(class_map['accessories'])}, {random.choice(class_map['footwear'])}, {random.choice(class_map['bag'])}, match nicely")

    elif 'bottom' in detected and 'accessories' in detected and not any(cls in detected for cls in ['top','fulldress','footwear','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['top'])}")
        suggestions.append(f"{random.choice(class_map['bag'])}, {random.choice(class_map['footwear'])}, {random.choice(class_map['hat'])}, match nicely")
    
    elif 'top' in detected and not any(cls in detected for cls in ['bottom','fulldress','footwear','accessories','bag','outerwear','hat']):
        suggestions.append(f"Pair with {', '.join(class_map['bottom'])}")
        footwear_item = random.choice(class_map['footwear'])
        accessories_item = random.choice(class_map['accessories'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"{footwear_item}, {accessories_item}, {bag_item}, {hat_item} match nicely")

    elif 'bottom' in detected and not any(cls in detected for cls in ['top','fulldress','footwear','accessories','bag','outerwear','hat']):
        suggestions.append(f"Style with {', '.join(class_map['top'])}")
        footwear_item = random.choice(class_map['footwear'])
        accessories_item = random.choice(class_map['accessories'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"{footwear_item}, {accessories_item}, {bag_item}, {hat_item} complement the look")
    # Rule for fulldress
    elif 'fulldress' in detected and not any(cls in detected for cls in ['top','bottom','footwear','accessories','bag','outerwear','hat']):
        suggestions.append(f"Accessorize with {', '.join(class_map['accessories'])}")
        footwear_item = random.choice(class_map['footwear'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"Pair with {footwear_item}, {bag_item}, and {hat_item} for balance")

    # Rule for footwear
    elif 'footwear' in detected and not any(cls in detected for cls in ['top','bottom','fulldress','accessories','bag','outerwear','hat']):
        suggestions.append(f"Match with tops like {', '.join(class_map['top'])} and bottoms like {', '.join(class_map['bottom'])}")
        accessories_item = random.choice(class_map['accessories'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"Add {accessories_item}, {bag_item}, and {hat_item} to complete the outfit")

    # Rule for accessories
    elif 'accessories' in detected and not any(cls in detected for cls in ['top','bottom','fulldress','footwear','bag','outerwear','hat']):
        suggestions.append(f"Works well with {', '.join(class_map['top'])} and {', '.join(class_map['bottom'])}, or even {', '.join(class_map['fulldress'])}")
        footwear_item = random.choice(class_map['footwear'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"Combine with {footwear_item}, {bag_item}, and {hat_item} for extra flair")

    # Rule for bag
    elif 'bag' in detected and not any(cls in detected for cls in ['top','bottom','fulldress','footwear','accessories','outerwear','hat']):
        suggestions.append(f"Complements tops like {', '.join(class_map['top'])} and bottoms like {', '.join(class_map['bottom'])}, or even {', '.join(class_map['fulldress'])}")
        footwear_item = random.choice(class_map['footwear'])
        accessories_item = random.choice(class_map['accessories'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"Style with {footwear_item}, {accessories_item}, and {hat_item} for variety")

    # Rule for outerwear
    elif 'outerwear' in detected and not any(cls in detected for cls in ['top','bottom','fulldress','footwear','accessories','bag','hat']):
        suggestions.append(f"Layer over {', '.join(class_map['top'])} with bottoms like {', '.join(class_map['bottom'])} and footwear such as {', '.join(class_map['footwear'])}")
        accessories_item = random.choice(class_map['accessories'])
        bag_item = random.choice(class_map['bag'])
        hat_item = random.choice(class_map['hat'])
        suggestions.append(f"Finish with {accessories_item}, {bag_item}, and {hat_item} for detail")

    # Rule for hat
    elif 'hat' in detected and not any(cls in detected for cls in ['top','bottom','fulldress','footwear','accessories','bag','outerwear']):
        suggestions.append(f"Pairs well with {', '.join(class_map['top'])}, {', '.join(class_map['bottom'])}, and outerwear like {', '.join(class_map['outerwear'])}")
        footwear_item = random.choice(class_map['footwear'])
        accessories_item = random.choice(class_map['accessories'])
        bag_item = random.choice(class_map['bag'])
        suggestions.append(f"Match with {footwear_item}, {accessories_item}, and {bag_item} for polish")

    return suggestions

def genai_style_advice(detected, detected_colors):
    """
    Build a prompt that includes both detected items and their colors.
    Calls OpenAI chat completion and returns the assistant text.
    """
    if not detected:
        return "No fashion items detected — please upload a clearer image."

    # Build prompt with both items and colors
    prompt_lines = []
    prompt_lines.append("You are a professional fashion stylist.")
    prompt_lines.append("I detected the following clothing items:")
    prompt_lines.append(", ".join(detected))
    if detected_colors:
        prompt_lines.append("\nDetected colors per item:")
        for item in detected:
            color = detected_colors.get(item)
            if color:
                prompt_lines.append(f"- {item.capitalize()}: {color}")
            else:
                prompt_lines.append(f"- {item.capitalize()}: (color not detected)")

    prompt_lines.append("\nTask: Suggest exactly two distinct outfit recommendations based on the detected items and colors.")
    prompt_lines.append("One outfit should be for men and one for women. Include footwear and accessories where relevant.")
    prompt_lines.append("Constraints:")
    prompt_lines.append("- Do not repeat phrases.")
    prompt_lines.append("- Keep each outfit concise (1–3 short sentences).")
    prompt_lines.append("- Output ONLY the two labeled outfits, nothing else.")
    prompt = "\n".join(prompt_lines)

    # Call OpenAI chat completion (uses your existing OpenAI client)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # keep your preferred model
            messages=[
                {"role": "system", "content": "You are a helpful and creative fashion stylist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"GenAI error: {e}"



# --- Zalando Scraper (via Apify) ---
def fetch_zalando_products(keyword: str, apify_token: str, max_items: int = 5):
    """
    Fetch Zalando products using Apify Zalando scraper.
    Docs: https://apify.com/ilong_mamman/zalando-scraper
    """
    run_url = f"https://api.apify.com/v2/acts/ilong_mamman~zalando-scraper/runs?token={apify_token}"
    input_data = {
        "startUrls": [
            {"url": f"https://www.zalando.co.uk/catalogue?search={keyword}"}
        ],
        "maxRequestsPerCrawl": max_items
    }
    run_resp = requests.post(run_url, json=input_data)
    if run_resp.status_code != 201:
        return []

    run_data = run_resp.json()
    dataset_id = run_data.get("defaultDatasetId")
    if not dataset_id:
        return []

    dataset_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={apify_token}"
    items_resp = requests.get(dataset_url)
    if items_resp.status_code == 200:
        return items_resp.json()
    return []


# --- Fallback dataset ---
def fetch_fallback_products():
    return [
        {
            "title": "Slim Fit Denim Jacket",
            "image": "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c",
            "price": "£49.99",
            "url": "https://example.com/denim-jacket"
        },
        {
            "title": "White Sneakers",
            "image": "https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f",
            "price": "£59.99",
            "url": "https://example.com/white-sneakers"
        }
    ]


# --- Display products ---
def show_products(products: list):
    """
    Display products in a nice card layout in Streamlit.
    Each product must have: title, price, currency, url, image
    """
    if not products:
        st.info("No products found.")
        return

    cols = st.columns(3)  # 3 products per row

    for i, product in enumerate(products):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 12px;
                    margin-bottom: 20px;
                    text-align: center;
                    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
                ">
                    <img src="{product.get('image', '')}" 
                         alt="Product Image" 
                         style="width:100%; height:200px; object-fit:cover; border-radius:10px; margin-bottom:10px;" />
                    <h4 style="color:#FFD700; font-size:16px; min-height:40px;">{product.get('title', 'No title')}</h4>
                    <p style="color:white; font-size:14px; margin:5px 0;">
                        💲 {product.get('price', 'N/A')} {product.get('currency', '')}
                    </p>
                    <a href="{product.get('url', '#')}" target="_blank" 
                       style="display:inline-block; padding:6px 12px; background:#FF69B4; color:white; 
                              border-radius:6px; text-decoration:none; font-weight:bold;">
                        View on Asos
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )

def genai_text_advice(user_text: str):
    """
    Generate AI-based fashion advice directly from user input text.
    """
    prompt = f"""
    The user is asking for fashion advice.
    Input: "{user_text}"
    Task: Suggest a complete outfit including clothing items, colors, and accessories.
    Format the output as a clear recommendation.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",   # or gpt-4.1 / gpt-3.5-turbo depending on your quota
        messages=[
            {"role": "system", "content": "You are a helpful and creative fashion stylist."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content
def extract_fashion_keywords(text: str) -> list:
    """
    Extract multiple relevant fashion keywords from AI advice or user text.
    """
    fashion_keywords = [
        "dress", "shirt", "tshirt", "t-shirt", "jeans", "pants", "trousers", "shorts",
        "skirt", "jacket", "coat", "blazer", "hoodie", "sweater", "cardigan",
        "sneakers", "shoes", "boots", "sandals", "heels", "loafers",
        "bag", "handbag", "clutch", "backpack", "scarf", "hat", "cap", "belt"
    ]

    found = []
    text_lower = text.lower()
    for word in fashion_keywords:
        if word in text_lower:
            found.append(word)

    return list(set(found)) if found else ["fashion"]

def extract_fashion_keywords_gpt(text: str) -> list:
    """
    Use GPT to extract fashion-related keywords from AI advice or user input.
    """
    prompt = f"""
    Extract the main fashion/clothing items from this text.
    Return them as a comma-separated list of single keywords (e.g. 'dress, heels, clutch').
    
    Text: "{text}"
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts fashion items from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=50
    )

    result = response.choices[0].message.content.strip()
    return [kw.strip() for kw in result.split(",") if kw.strip()]

# EBAY_SANDBOX_APP_ID = "Jithinku-fashiona-SBX-a1152cb22-cce8dd30"
EBAY_SANDBOX_APP_ID = "Jithinku-fashiona-PRD-54394c22d-a3217260"
def fetch_ebay_sandbox_products(keyword: str, limit: int = 5):
    """
    Fetch products from eBay Sandbox by keyword.
    Returns a list of dicts with: title, price, currency, url, image
    """
    url = "https://svcs.sandbox.ebay.com/services/search/FindingService/v1"
    params = {
        "OPERATION-NAME": "findItemsByKeywords",
        "SERVICE-VERSION": "1.0.0",
        "SECURITY-APPNAME": EBAY_SANDBOX_APP_ID,
        "RESPONSE-DATA-FORMAT": "JSON",
        "REST-PAYLOAD": "",
        "keywords": keyword,
        "paginationInput.entriesPerPage": str(limit)
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        items = (
            data.get("findItemsByKeywordsResponse", [])[0]
            .get("searchResult", [])[0]
            .get("item", [])
        )

        results = []
        for item in items:
            results.append({
                "title": item.get("title", ["N/A"])[0],
                "price": item["sellingStatus"][0]["currentPrice"][0]["__value__"],
                "currency": item["sellingStatus"][0]["currentPrice"][0]["@currencyId"],
                "url": item.get("viewItemURL", ["#"])[0],
                "image": item.get("galleryURL", [""])[0]
            })

        return results

    except Exception as e:
        print("❌ Error fetching from eBay Sandbox:", e)
        return []

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

def fetch_asos_products(keyword: str, limit: int = 5):
    """
    Fetch products from ASOS API via RapidAPI
    """
    if not RAPIDAPI_KEY:
        st.error("Missing RAPIDAPI_KEY. Please set it in your .env file.")
        return []

    url = "https://asos2.p.rapidapi.com/products/v2/list"
    querystring = {
        "q": keyword,
        "store": "US",
        "offset": "0",
        "limit": str(limit),
        "categoryId": "4209",   # 4209 = Clothing
        "country": "US",
        "sort": "freshness",
        "currency": "USD",
        "sizeSchema": "US",
        "lang": "en-US"
    }

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,          # ✅ loaded automatically
        "x-rapidapi-host": "asos2.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        products = []
        for item in data.get("products", []):
            products.append({
                "title": item.get("name"),
                "price": item.get("price", {}).get("current", {}).get("text", "N/A"),
                "image": f"https://{item.get('imageUrl')}" if item.get("imageUrl") else None,
                "url": f"https://www.asos.com/{item.get('url')}" if item.get("url") else None
            })
        return products
    else:
        st.error(f"ASOS API error: {response.status_code}")
        return []

def expand_fashion_keywords(advice: str, max_keywords: int = 10):
    """
    Expands the fashion advice into multiple precise product search keywords.
    """
    prompt = f"""
    Extract product search keywords from this fashion advice:

    {advice}

    Return ONLY a JSON array of up to {max_keywords} keyword phrases.
    Each keyword phrase should be 2–4 words long and useful for shopping searches.
    Example:
    ["white linen short sleeve shirt", "pastel floral sundress", "tan strappy sandals"]
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an assistant that generates precise fashion product search keywords."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.4
    )

    raw_output = response.choices[0].message.content.strip()
    
    try:
        # Force JSON parsing
        keywords = json.loads(raw_output)
        if isinstance(keywords, list):
            return keywords
    except Exception:
        # fallback: try to clean and extract JSON array manually
        start = raw_output.find("[")
        end = raw_output.rfind("]") + 1
        if start != -1 and end != -1:
            try:
                return json.loads(raw_output[start:end])
            except:
                pass

    return []


def render_colors_in_text(text: str) -> str:
    """
    Finds hex color codes (#xxxxxx) in the text and replaces them
    with a colored square + the hex code.
    """
    def repl(match):
        hex_code = match.group(0)
        return f"""<span style="display:inline-block;
                               width:14px;
                               height:14px;
                               background:{hex_code};
                               border:1px solid #ccc;
                               border-radius:3px;
                               margin-right:4px;"></span><code>{hex_code}</code>"""

    return re.sub(r"#(?:[0-9a-fA-F]{3}){1,2}", repl, text)


