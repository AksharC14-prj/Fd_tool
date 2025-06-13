import math
import streamlit as st
import pandas as pd
import os
import uuid
from PIL import Image
import shutil
import google.generativeai as genai

st.set_page_config(
    page_title="Food Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'classification_done' not in st.session_state:
    st.session_state.classification_done = False
if 'save_complete' not in st.session_state:
    st.session_state.save_complete = False
if 'page' not in st.session_state:
    st.session_state.page = 1
if 'div_selections' not in st.session_state:
    st.session_state.div_selections = {}

def reset_app():
    """Reset all session state variables to start a new classification"""
    for key in ['processed_results', 'processing_complete', 'classification_done', 'save_complete', 'div_selections']:
        if key in st.session_state:
            if key == 'processed_results':
                st.session_state[key] = []
            elif key == 'div_selections':
                st.session_state[key] = {}
            else:
                st.session_state[key] = False
    st.session_state.page = 1
    # Clear temp directories if they exist
    try:
        if os.path.exists("temp"):
            shutil.rmtree("temp")
        if os.path.exists("resized_images"):
            shutil.rmtree("resized_images")
        os.makedirs("temp", exist_ok=True)
    except Exception as e:
        st.error(f"Error cleaning up: {e}")

def update_match(i, option):
    """Update match status when selectbox selection changes"""
    if option == "-- Unmatched --":
        st.session_state.processed_results[i]["new_matched"] = None
    else:
        st.session_state.processed_results[i]["new_matched"] = option

def update_div_selection(image_id, item_name, div_value):
    """Update div selection for an image"""
    # Initialize the div selections for this item if not already done
    if item_name not in st.session_state.div_selections:
        st.session_state.div_selections[item_name] = {}
    
    # If a previous selection exists for this image, remove it
    old_div = None
    for div, img_id in st.session_state.div_selections[item_name].items():
        if img_id == image_id:
            old_div = div
            break
    
    if old_div:
        del st.session_state.div_selections[item_name][old_div]
    
    # Add the new selection
    if div_value != "-- Select --":
        st.session_state.div_selections[item_name][div_value] = image_id
    
    # Update the div_number in processed_results
    for i, res in enumerate(st.session_state.processed_results):
        if res["id"] == image_id:
            st.session_state.processed_results[i]["div_number"] = None if div_value == "-- Select --" else div_value

st.title("Food Classifier")
st.markdown("Upload your menu and images, then let the agent map each image to the closest menu item.")
st.info("⚠️ Reload the page to start a new classification using different files; Reset button in the sidebar is to clear all results.")

API_KEY = st.text_input(
    "Enter your Google Gemini API Key",
    type="password",
    help="Enter your Google Gemini API key",
    key="api_key"
)

with st.expander("ℹ️ How to set up Google Gemini API Key"):
    st.write("""
    To access this tool, you'll need to provide a Google Gemini API Key. Here's how to obtain one:

    1. Visit the Google AI for Developers website: https://ai.google.dev/
    2. Click on the "Explore in Google AI Studio" button
    3. You'll be redirected to the API key management page (https://aistudio.google.com/apikey)
    4. Click on "Create API Key"
    5. Select an existing Google project from the dropdown menu (or create a new one if needed)
    6. Click "Create" to generate your API key
    7. Copy your newly created API key and paste it into the text box above
    """)

with st.sidebar:
    st.header("Configuration")
    st.subheader("1. Data Input")
    uploaded_excel = st.file_uploader(
        "Upload Menu Excel",
        type=["xlsx"],
        key="uploader_excel"
    )    
    image_folder = st.text_input(
        "Path to Images Folder",
        help="Folder containing your dish images.",
        key="sidebar_image_folder"
    )    
    uploaded_images = st.file_uploader(
        "Or Upload Dish Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="uploader_images"
    )
    st.subheader("2. Output Settings")
    output_directory = st.text_input(
        "Output Folder Path",
        help="Where to save processed images and results",
        key="output_directory"
    )
    if st.button("Reset"):
        reset_app()
        st.rerun()
    st.subheader("3. Resize pixel size")
    image_size = st.number_input(
        "Resize Image to (px)",
        min_value=32,
        max_value=3048,
        step=32,
        value=256
    )

if st.session_state.processing_complete:
    total_items = len(st.session_state.processed_results)
    matched_count = sum(1 for res in st.session_state.processed_results if res.get("new_matched"))
    unmatched_count = total_items - matched_count
    st.sidebar.subheader("Classification Summary")
    st.sidebar.write(f"✅ Matched: {matched_count}")
    st.sidebar.write(f"❌ Unmatched: {unmatched_count}")
    st.sidebar.write(f"📊 Total Images: {total_items}")

start = False
if not (st.session_state.processing_complete or st.session_state.save_complete):
    start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
    with start_col2:
        if not uploaded_excel:
            st.warning("Please upload a menu Excel file to continue")
        elif not API_KEY:
            st.warning("Please enter your API key to continue") 
        elif not (image_folder or uploaded_images):
            st.warning("Please provide images via folder path or upload")
        elif not output_directory:
            st.warning("Please provide output folder path")
        else:
            start = st.button(
                "🚀 Start Classification",
                key="btn_start_classification"
            )

def resize_image(img, size=image_size):
    """Resize image to a square with specified dimensions"""
    square_img = Image.new('RGB', (size, size), (255, 255, 255))
    new_width = size
    new_height = size
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    paste_x = (size - new_width) // 2
    paste_y = (size - new_height) // 2
    square_img.paste(resized_img, (paste_x, paste_y))
    return square_img

if uploaded_excel:
    try:
        df = pd.read_excel(uploaded_excel)
        item_names = df['ItemName'].tolist()
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        st.stop()
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
    folder_images = []
    if image_folder and os.path.isdir(image_folder):
        folder_images = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith(valid_exts)
        ]
    combined_images = folder_images.copy()
    if uploaded_images:
        os.makedirs("temp", exist_ok=True)
        for file in uploaded_images:
            temp_path = os.path.join("temp", file.name)
            with open(temp_path, "wb") as f:
                f.write(file.read())
            combined_images.append(temp_path)

    if not combined_images:
        st.warning("❌ No images provided. Please upload or select a folder.")
    elif start and not st.session_state.processing_complete:
        if not API_KEY:
            st.error("API Key is required. Please provide your Google Gemini API key.")
            st.stop()
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        total = len(combined_images)
        progress = st.progress(0)
        resized_folder = os.path.join(os.getcwd(), "resized_images")
        os.makedirs(resized_folder, exist_ok=True)
        batch_size = 20
        counter = 0
        for start_idx in range(0, total, batch_size):
            batch_paths = combined_images[start_idx:start_idx + batch_size]
            batch_imgs = []
            for p in batch_paths:
                img = Image.open(p)
                resized = resize_image(img, size=image_size)
                fname = os.path.basename(p)
                save_path = os.path.join(resized_folder, f"resized_{fname}")
                resized.save(save_path)
                batch_imgs.append(resized)
            batch_filenames = [os.path.basename(p) for p in batch_paths]
            prompt = (
                f"I'm sending you {len(batch_imgs)} images and my menu: {item_names}.\n"
                f"The image filenames are: {batch_filenames}.\n"
                "For each image, return ONLY the dish name that best matches a menu item.\n"
                "If the filename matches or resembles a menu item, prioritize it based on image; otherwise, ignore the filename and predict based on the image.\n"
                "Output exactly one name per line in the same order as the images."
                "**return only predicted dish name, nothing else**"
            )
            with st.spinner(f"Processing images {start_idx+1}–{start_idx+len(batch_paths)} of {total}"):
                resp = model.generate_content([*batch_imgs, prompt])
                lines = resp.text.strip().splitlines()
            for path, resized, pred in zip(batch_paths, batch_imgs, lines):
                image_id = uuid.uuid4().hex
                matched = pred.strip()
                matched_index = next((i for i, name in enumerate(item_names) if name.lower() == matched.lower()), -1)
                new_filename = f"{uuid.uuid4().hex}{os.path.splitext(path)[1]}"
                st.session_state.processed_results.append({
                    "id": image_id,
                    "original_path": path,
                    "path": os.path.join(resized_folder, f"resized_{os.path.basename(path)}"),
                    "filename": os.path.basename(path),
                    "image": resized,
                    "predicted": matched,
                    "matched": matched,
                    "matched_index": matched_index,
                    "new_matched": matched,
                    "new_filename": new_filename,
                    "div_number": None  # Initialize div_number as None
                })
                counter += 1
                progress.progress(counter / total)

        st.session_state.processing_complete = True
        st.session_state.page = 1 
        st.rerun()
    
    if st.session_state.processing_complete and not st.session_state.classification_done:
        st.success("🎉 Classification completed! Review and adjust matches below.")
        total_items = len(st.session_state.processed_results)
        items_per_page = 25
        total_pages = math.ceil(total_items / items_per_page)
        start_idx = (st.session_state.page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, len(st.session_state.processed_results))
        for i in range(start_idx, end_idx):
            res = st.session_state.processed_results[i]
            c1, c2, c3 = st.columns([2, 3, 1])
            with c1:
                st.image(res["image"], caption=res["filename"], use_column_width=True)
            with c2:
                st.write(f"**Predicted:** {res['predicted']}")
                opts = ["-- Unmatched --"] + item_names
                if res["new_matched"]:
                    try:
                        default = item_names.index(res["new_matched"]) + 1
                    except ValueError:
                        default = 0
                elif res["matched_index"] >= 0:
                    default = res["matched_index"] + 1
                else:
                    default = 0
                new_selected = st.selectbox(
                    f"Match for image {res['filename']}:",
                    opts,
                    index=default,
                    key=f"select_{i}",
                    on_change=update_match,
                    args=(i, st.session_state.get(f"select_{i}")),
                    label_visibility="collapsed"  # Hide label for compactness
                )
                if new_selected == "-- Unmatched --":
                    st.session_state.processed_results[i]["new_matched"] = None
                else:
                    st.session_state.processed_results[i]["new_matched"] = new_selected
            
            with c3:
                status = "✅" if st.session_state.processed_results[i]["new_matched"] else "❌"
                if status == "✅":
                    st.success(status)
                else:
                    st.warning(status)
            st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if (st.session_state.page > 1):
                if st.button("← Previous", key="prev_page") and st.session_state.page > 1:
                    st.session_state.page -= 1
                    st.rerun()
        with col2:
            st.markdown(f"<p style='text-align:center'>Page {st.session_state.page} of {total_pages}</p>", unsafe_allow_html=True)
        with col3:
            if st.session_state.page < total_pages:  
                if st.button("Next →", key="next_page"):
                    st.session_state.page += 1
                    st.rerun()
        
        # Add "Classification Done" button instead of Save
        classification_done = st.button(
            "✓ Classification Done",
            key="btn_classification_done"
        )
        
        if classification_done:
            st.session_state.classification_done = True
            st.rerun()
            
    elif st.session_state.classification_done and not st.session_state.save_complete:
        st.success("🎉 Classification completed! Now select div number for each matched image.")
        
        # Group images by item name
        item_images = {}
        for res in st.session_state.processed_results:
            if res.get("new_matched"):
                item_name = res.get("new_matched")
                if item_name not in item_images:
                    item_images[item_name] = []
                item_images[item_name].append(res)
        
        # Auto-assign div numbers only for items with exactly 1 image
        for item_name, images in item_images.items():
            if len(images) == 1:
                # Automatically map to div_1
                if item_name not in st.session_state.div_selections:
                    st.session_state.div_selections[item_name] = {}
                
                # Single image to div_1
                st.session_state.div_selections[item_name]["div_1"] = images[0]["id"]
                # Update the processed_results with div_number
                for i, res in enumerate(st.session_state.processed_results):
                    if res["id"] == images[0]["id"]:
                        st.session_state.processed_results[i]["div_number"] = "div_1"
        
        # Sort the results by item name (ascending)
        sorted_results = []
        for item_name in sorted(item_names):
            for res in st.session_state.processed_results:
                if res.get("new_matched") == item_name:
                    sorted_results.append(res)
        
        # Add any unmatched items at the end
        for res in st.session_state.processed_results:
            if not res.get("new_matched"):
                sorted_results.append(res)
        
        # Display results with div selection
        total_items = len(sorted_results)
        items_per_page = 25
        total_pages = math.ceil(total_items / items_per_page)
        start_idx = (st.session_state.page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        for i in range(start_idx, end_idx):
            res = sorted_results[i]
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            with c1:
                st.image(res["image"], caption=res["filename"], use_column_width=True)
            with c2:
                item_name = res.get("new_matched")
                if item_name:
                    st.write(f"**Matched Item:** {item_name}")
                    
                    # Get already used div numbers for this item
                    used_divs = []
                    if item_name in st.session_state.div_selections:
                        used_divs = list(st.session_state.div_selections[item_name].keys())
                    
                    # Create div options (div_1, div_3, div_4, div_5, div_6) - excluding div_2
                    div_options = ["-- Select --"]
                    for d in [1, 3, 4, 5, 6]:  # Exclude div_2
                        div_option = f"div_{d}"
                        # Add option only if not already used for another image of this item
                        if div_option not in used_divs or (res["div_number"] == div_option):
                            div_options.append(div_option)
                    
                    # Set default index based on current selection
                    default_idx = 0
                    if res["div_number"] in div_options:
                        default_idx = div_options.index(res["div_number"])
                    
                    div_selected = st.selectbox(
                        f"Select div for {res['filename']}:",
                        div_options,
                        index=default_idx,
                        key=f"div_select_{res['id']}"
                    )
                    update_div_selection(res['id'], item_name, div_selected)

                else:
                    st.warning("**Unmatched**")
            
            with c3:
                if res.get("new_matched"):
                    div_status = "✅" if res.get("div_number") else "❓"
                    if div_status == "✅":
                        st.success(f"Div: {res.get('div_number')}")
                    else:
                        st.warning("No div selected")
                else:
                    st.warning("Unmatched - No div needed")
            
            st.markdown("---")
        
        # Pagination controls
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if (st.session_state.page > 1):
                if st.button("← Previous", key="prev_page_div") and st.session_state.page > 1:
                    st.session_state.page -= 1
                    st.rerun()
        with col2:
            st.markdown(f"<p style='text-align:center'>Page {st.session_state.page} of {total_pages}</p>", unsafe_allow_html=True)
        with col3:
            if st.session_state.page < total_pages:  
                if st.button("Next →", key="next_page_div"):
                    st.session_state.page += 1
                    st.rerun()
        
        # Check if all matched images have div selections
        all_matched_have_div = True
        missing_div_count = 0
        for res in st.session_state.processed_results:
            if res.get("new_matched") and not res.get("div_number"):
                all_matched_have_div = False
                missing_div_count += 1
        
        # Now show the save button only if all matched images have div selections
        if all_matched_have_div:
            save_clicked = st.button(
                "💾 Save Results",
                key="btn_save_results"
            )
        else:
            st.warning(f"⚠️ Cannot save results yet. {missing_div_count} matched image(s) still need div selections.")
            # Display disabled button visually
            st.markdown(
                """
                <button class="stButton" style="opacity: 0.6; cursor: not-allowed;" disabled>
                    💾 Save Results
                </button>
                """, 
                unsafe_allow_html=True
            )
            save_clicked = False
        
        if save_clicked:
            # Use the output directory from user input
            save_dir = st.session_state.output_directory if st.session_state.output_directory else os.path.join(os.getcwd(), "Sorted_Items")
            os.makedirs(save_dir, exist_ok=True)
            unmatched_dir = os.path.join(save_dir, "unmatched")
            os.makedirs(unmatched_dir, exist_ok=True)
            
            # Create SQL file
            sql_file_path = os.path.join(save_dir, "insert_statements.txt")
            unmatched_list = []
            saved_count = 0
            unmatched_saved_count = 0
            duplicate_div2_count = 0
            final_item_matches = {}
            
            for idx, row in df.iterrows():
                item_number = str(row['ItemNumber'])
                item_name = row['ItemName']
                final_item_matches[item_name] = {
                    "ItemNumber": item_number,
                    "ItemName": item_name,
                    "Matched_images_original_names": [],
                    "modified_image_names": [],
                    "Number_of_Matches": 0
                }
            
            with open(sql_file_path, 'w') as sql_file:
                with st.spinner("Saving matched and unmatched images..."):
                    for res in st.session_state.processed_results:
                        new_fname = res["new_filename"]
                        
                        if res.get("new_matched") and res["new_matched"] in final_item_matches:
                            row = df[df['ItemName'] == res["new_matched"]]
                            if not row.empty:
                                item_number = str(row.iloc[0]['ItemNumber'])
                                
                                resize_folder = os.path.join(save_dir, "resize", item_number)
                                original_folder = os.path.join(save_dir, "original", item_number)
                                os.makedirs(resize_folder, exist_ok=True)
                                os.makedirs(original_folder, exist_ok=True)

                                try:
                                    resized_dest = os.path.join(resize_folder, new_fname)
                                    shutil.copy2(res["path"], resized_dest)
                                    original_dest = os.path.join(original_folder, new_fname)
                                    shutil.copy2(res["original_path"], original_dest)
                                    saved_count += 1
                                    item = final_item_matches[res["new_matched"]]
                                    item["Matched_images_original_names"].append(res["filename"])
                                    item["modified_image_names"].append(new_fname)
                                    item["Number_of_Matches"] += 1
                                    
                                    # Use the selected div number (or div_1 as default if none selected)
                                    div_number = res.get("div_number") or "div_1"
                                    sort_order = div_number.split("_")[1] if "_" in div_number else "1"
                                    
                                    # Extract the file extension for mime_type
                                    file_ext = os.path.splitext(new_fname)[1][1:].lower()
                                    mime_type = f"img/{file_ext}"
                                    if file_ext == "jpg":
                                        mime_type = "image/jpeg"
                                    
                                    # Format the SQL query with the custom div number
                                    sql_statement = (
                                        f"INSERT INTO `mh_media` (`id`, `entity_type`, `entity_id`, `file_name`, "
                                        f"`mime_type`, `sort_order`) VALUES ('{os.path.splitext(new_fname)[0]}', "
                                        f"'{div_number}', '{item_number}', '{new_fname}', "
                                        f"'{mime_type}', '{sort_order}');\n"
                                    )
                                    sql_file.write(sql_statement)

                                    # Generate div_2 duplicate for div_1 images
                                    if div_number == "div_1":
                                        # Create a new UUID for div_2
                                        div2_uuid = uuid.uuid4().hex
                                        div2_filename = f"{div2_uuid}{os.path.splitext(new_fname)[1]}"
                                        
                                        # Copy the same image file with the new UUID
                                        div2_resized_dest = os.path.join(resize_folder, div2_filename)
                                        shutil.copy2(res["path"], div2_resized_dest)
                                        div2_original_dest = os.path.join(original_folder, div2_filename)
                                        shutil.copy2(res["original_path"], div2_original_dest)
                                        
                                        # Add SQL statement for div_2
                                        div2_sql_statement = (
                                            f"INSERT INTO `mh_media` (`id`, `entity_type`, `entity_id`, `file_name`, "
                                            f"`mime_type`, `sort_order`) VALUES ('{div2_uuid}', "
                                            f"'div_2', '{item_number}', '{div2_filename}', "
                                            f"'{mime_type}', '2');\n"
                                        )
                                        sql_file.write(div2_sql_statement)
                                        duplicate_div2_count += 1

                                except Exception as e:
                                    st.error(f"Error saving {res['filename']}: {e}")
                        else:
                            try:
                                unmatched_resize = os.path.join(unmatched_dir, "resize")
                                unmatched_original = os.path.join(unmatched_dir, "original")
                                os.makedirs(unmatched_resize, exist_ok=True)
                                os.makedirs(unmatched_original, exist_ok=True)
                                resized_dest = os.path.join(unmatched_resize, new_fname)
                                shutil.copy2(res["path"], resized_dest)
                                original_dest = os.path.join(unmatched_original, new_fname)
                                shutil.copy2(res["original_path"], original_dest)
                                unmatched_saved_count += 1
                                unmatched_list.append({
                                    "Image File": res["filename"],
                                    "New Filename": new_fname, 
                                    "Prediction": res["predicted"]
                                })
                            except Exception as e:
                                st.error(f"Error saving unmatched image {res['filename']}: {e}")
                                
            summary_df = pd.DataFrame(list(final_item_matches.values()))
            summary_df["Matched_images_original_names"] = summary_df["Matched_images_original_names"].apply(lambda x: ", ".join(x))
            summary_df["modified_image_names"] = summary_df["modified_image_names"].apply(lambda x: ", ".join(x))
            st.subheader("✅ Final Mapped Items (All Menu Items)")
            st.dataframe(summary_df)
            st.download_button(
                "Download Final Mapped Items CSV",
                summary_df.to_csv(index=False).encode(),
                file_name="final_mapped_items.csv",
                mime="text/csv"
            )
            
            st.success(f"✓ Saved {saved_count} matched images and {unmatched_saved_count} unmatched images to `{save_dir}`")
            st.success(f"✓ Generated {duplicate_div2_count} div_2 duplicates of div_1 images")
            st.success(f"✓ SQL insert statements saved to `{sql_file_path}`")
            
            
            if unmatched_list:
                df_un = pd.DataFrame(unmatched_list)
                st.download_button(
                    "Download Unmatched CSV",
                    df_un.to_csv(index=False).encode(),
                    file_name="unmatched.csv",
                    mime="text/csv"
                )
            st.session_state.save_complete = True
        
        if st.button(
            "Done",
            key="btn_restart1"
        ):
            reset_app()
            st.rerun()
    
    elif st.session_state.save_complete:
        st.success("🎉 Done and saved!")
        if st.button(
            "Start New Classification",
            key="btn_restart2"
        ):
            reset_app()
            st.rerun()
    
    elif not st.session_state.processing_complete:
        st.info("📂 Configure all inputs and click the **Start Classification** button.")
else:
    st.info("Awaiting menu Excel upload in sidebar...")
