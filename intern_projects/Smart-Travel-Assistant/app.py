import streamlit as st
from datetime import date
import uuid
from utils import llm_suggest_items
from db import get_travel_info, get_checklist, save_travel_details, save_checklist
from email_utils import send_welcome_email, send_checklist_update_email

st.set_page_config(page_title="Smart Travel Assistant", layout="centered")
st.title("🧳 Smart Travel Assistant")

# Check if returning user
st.markdown("### 🔑 Already registered?")
user_id = st.text_input("Enter your User ID")

if user_id:
    travel = get_travel_info(user_id)
    checklist = get_checklist(user_id)

    if not travel:
        st.error("User ID not found. Please register as a new user.")
    else:
        st.success(f"Welcome back, {travel['name']}!")

        # Edit travel details
        st.markdown("#### ✏️ Update Travel Details")
        name = st.text_input("Your Name", value=travel["name"])
        email = st.text_input("Email", value=travel["email"])
        destination = st.text_input("Destination", value=travel["destination"])
        purpose = st.text_input("Purpose of Travel", value=travel["purpose"])
        companions = st.text_input("Companions (comma-separated)", value=", ".join(travel["companions"]))
        start_date = st.date_input("Start Date", value=date.fromisoformat(travel["startDate"]))
        end_date = st.date_input("End Date", value=date.fromisoformat(travel["endDate"]))

        if st.button("Update Details"):
            updated_data = {
                "userId": user_id,
                "name": name,
                "email": email,
                "destination": destination,
                "purpose": purpose,
                "companions": [c.strip() for c in companions.split(",")],
                "startDate": str(start_date),
                "endDate": str(end_date)
            }
            save_travel_details(updated_data)
            st.success("Travel details updated!")

        st.markdown("#### 🧾 Update Checklist")
        items = [i["item"] for i in checklist["items"]]
        priority = [i["item"] for i in checklist["items"] if i.get("priority")]

        # Initial checklist selection
        base_selected_items = st.multiselect("Checklist (edit or remove)", items, default=items)
        priority_items = st.multiselect("Mark Priority Items", base_selected_items, default=[p for p in priority if p in base_selected_items])

        st.markdown("#### ➕ Add More Items")
        # Only generate AI suggestions once and store in session
        if "update_ai_recs" not in st.session_state:
            st.session_state.update_ai_recs = llm_suggest_items(destination, purpose, items)
        ai_recs = st.session_state.update_ai_recs
        st.markdown("Enter numbers for AI suggestions to add (comma-separated):")
        for i, rec in enumerate(ai_recs):
            st.write(f"{i+1}. {rec}")
        rec_indexes_input = st.text_input("Which suggestions do you want to add?")
        added_recs = []
        if rec_indexes_input:
            try:
                indexes = [int(i.strip())-1 for i in rec_indexes_input.split(",") if i.strip().isdigit()]
                added_recs = [ai_recs[i] for i in indexes if 0 <= i < len(ai_recs)]
            except:
                st.warning("Invalid selection. Please enter comma-separated numbers.")
        manual_input = st.text_input("Add custom items (comma-separated)")
        manual_items = [i.strip() for i in manual_input.split(",") if i.strip()]

        # Combine selected items + new ones from suggestions + manual entries
        updated_items = list(set(base_selected_items + added_recs + manual_items))

        # Combine again with latest added ones
        final_items = list(set(updated_items))
        final_selected_items = st.multiselect("Final Checklist (after additions)", final_items, default=final_items)
        priority_items = st.multiselect("Mark Priority Items Again", final_selected_items, default=[p for p in priority_items if p in final_selected_items])

        
        if st.button("Confirm Updated Checklist"):
            updated_list = [{"item": i, "priority": i in priority_items} for i in final_selected_items]
            save_checklist(user_id, updated_list)
            send_checklist_update_email(email, updated_list, name)
            st.success("Checklist updated and email sent!")
            st.success("Checklist updated and email sent!")

        if st.button("🔙 Back to Homepage"):
            st.experimental_rerun()

else:
    st.markdown("### 🆕 New User Registration")

    with st.form("trip_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Email")
        destination = st.text_input("Destination")
        purpose = st.text_input("Purpose of Travel")
        companions = st.text_input("Companions (comma-separated)").split(",")
        start_date = st.date_input("Start Date", min_value=date.today())
        end_date = st.date_input("End Date", min_value=start_date)
        initial_items = st.text_input("Initial packing items (comma-separated)").split(",")

        submitted = st.form_submit_button("Next")

    if submitted:
        destination = destination.strip()
        purpose = purpose.strip()
        if len(destination) < 3 or len(purpose) < 3:
            st.error("Destination and Purpose must each be at least 3 characters long.")
            st.stop()

        user_id = "U" + uuid.uuid4().hex[:6].upper()
        initial_items = [i.strip() for i in initial_items if i.strip()]
        st.session_state.user_id = user_id
        st.session_state.name = name
        st.session_state.email = email
        st.session_state.destination = destination
        st.session_state.purpose = purpose
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.companions = companions
        st.session_state.initial_items = initial_items
        st.session_state.recommendations = llm_suggest_items(destination, purpose, initial_items)

    if "user_id" in st.session_state:
        st.markdown("#### 📦 Add More Items")

        selected_recs = st.multiselect("Select from AI Suggestions", st.session_state.recommendations)
        manual_items_input = st.text_input("Add custom items (comma-separated)")
        manual_items = [i.strip() for i in manual_items_input.split(",") if i.strip()]

        combined_items = list(set(st.session_state.initial_items + selected_recs + manual_items))

        st.markdown("### ✅ Finalize Your Checklist")
        selected_items = st.multiselect("Final Checklist", combined_items, default=combined_items)
        priority_items = st.multiselect("Mark Priority Items", selected_items)

        if st.button("Confirm Checklist"):
            checklist = [{"item": i, "priority": i in priority_items} for i in selected_items]
            user_data = {
                "userId": st.session_state.user_id,
                "name": st.session_state.name,
                "email": st.session_state.email,
                "destination": st.session_state.destination,
                "purpose": st.session_state.purpose,
                "companions": [c.strip() for c in st.session_state.companions],
                "startDate": str(st.session_state.start_date),
                "endDate": str(st.session_state.end_date)
            }
            save_travel_details(user_data)
            save_checklist(st.session_state.user_id, checklist)
            send_welcome_email(
                user_data["email"], user_data["name"], user_data["destination"],
                user_data["startDate"], user_data["endDate"], user_data["purpose"],
                user_data["companions"], checklist, user_data["userId"]
            )

            send_checklist_update_email(user_data["email"], checklist, user_data["name"])
            st.success(f"Checklist confirmed and emails sent! Your User ID is: {st.session_state.user_id}")

            if st.button("🔙 Back to Homepage"):
                st.experimental_rerun()
