"""
History Viewer Component
View and manage analysis history
"""

import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd
from config import PATHS


def load_history():
    """Load analysis history from JSON file"""
    history_file = 'history/analysis_history.json'

    if not os.path.exists(history_file):
        return []

    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        return history
    except:
        return []


def clear_history():
    """Clear all history"""
    history_file = 'history/analysis_history.json'

    if os.path.exists(history_file):
        os.remove(history_file)

    if 'history' in st.session_state:
        st.session_state.history = []


def delete_entry(entry_id):
    """Delete a specific history entry"""
    history = load_history()
    history = [h for h in history if h.get('id') != entry_id]

    history_file = 'history/analysis_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)


def render(models):
    """Render the history viewer page"""

    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2c3e50;'>Analysis History</h1>
        <p style='color: #7f8c8d; font-size: 1.1em;'>View and manage past analyses</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Load history
    history = load_history()

    if not history:
        st.info("""
        **No history yet**

        Your analysis history will appear here once you start analyzing articles.

        Try analyzing an article in the **Single Article Analysis** page and save it to history!
        """)
        return

    # Statistics
    st.markdown("### Overview")

    fake_count = sum(1 for h in history if h.get('consensus') == 0)
    true_count = sum(1 for h in history if h.get('consensus') == 1)
    total = len(history)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Analyses", total)

    with col2:
        st.metric("True News", true_count)

    with col3:
        st.metric("Fake News", fake_count)

    with col4:
        fake_pct = (fake_count / total * 100) if total > 0 else 0
        st.metric("Fake %", f"{fake_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization
    from components import visualizations
    visualizations.plot_history_timeline(history)

    st.markdown("---")

    # Filters
    st.markdown("### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        filter_result = st.selectbox(
            "Filter by result",
            ["All", "True News", "Fake News"]
        )

    with col2:
        filter_agreement = st.selectbox(
            "Filter by agreement",
            ["All", "Unanimous (4/4)", "Strong (3/4)", "Split (2/4)"]
        )

    with col3:
        sort_order = st.selectbox(
            "Sort by",
            ["Newest First", "Oldest First"]
        )

    # Apply filters
    filtered_history = history.copy()

    if filter_result == "True News":
        filtered_history = [h for h in filtered_history if h.get('consensus') == 1]
    elif filter_result == "Fake News":
        filtered_history = [h for h in filtered_history if h.get('consensus') == 0]

    if filter_agreement == "Unanimous (4/4)":
        filtered_history = [h for h in filtered_history if h.get('agreement_count') == 4]
    elif filter_agreement == "Strong (3/4)":
        filtered_history = [h for h in filtered_history if h.get('agreement_count') == 3]
    elif filter_agreement == "Split (2/4)":
        filtered_history = [h for h in filtered_history if h.get('agreement_count') == 2]

    # Sort
    filtered_history = sorted(
        filtered_history,
        key=lambda x: x.get('timestamp', ''),
        reverse=(sort_order == "Newest First")
    )

    st.info(f"Showing {len(filtered_history)} of {total} entries")

    st.markdown("---")

    # Display history entries
    st.markdown("### History Entries")

    if not filtered_history:
        st.warning("No entries match your filters")
    else:
        for idx, entry in enumerate(filtered_history):
            consensus = entry.get('consensus', 0)
            agreement = entry.get('agreement_count', 0)
            timestamp = entry.get('timestamp', '')
            text_preview = entry.get('text_preview', 'No preview available')
            entry_id = entry.get('id', '')

            # Parse timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                time_str = timestamp

            # Create expandable entry
            with st.expander(f"**#{idx+1}** - {time_str} - {'VERIFIED' if consensus == 1 else 'UNVERIFIED'} ({agreement}/4 agree)"):
                st.markdown("**Article Preview:**")
                st.text(text_preview)

                st.markdown("**Model Predictions:**")

                if 'model_details' in entry and entry['model_details']:
                    cols = st.columns(4)

                    for col, detail in zip(cols, entry['model_details']):
                        with col:
                            pred = detail.get('prediction', 0)
                            conf = detail.get('confidence', 0)
                            name = detail.get('name', 'Unknown')

                            if pred == 1:
                                st.success(f"**{name}**\n\nTRUE\n\n{conf:.1f}%")
                            else:
                                st.error(f"**{name}**\n\nFAKE\n\n{conf:.1f}%")

                # Delete button
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button(f"Delete", key=f"delete_{entry_id}"):
                        delete_entry(entry_id)
                        st.rerun()

    st.markdown("---")

    # Export and management
    st.markdown("### Export & Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Export as CSV
        if filtered_history:
            df_data = []
            for entry in filtered_history:
                df_data.append({
                    'Timestamp': entry.get('timestamp', ''),
                    'Result': 'TRUE NEWS' if entry.get('consensus') == 1 else 'FAKE NEWS',
                    'Agreement': f"{entry.get('agreement_count', 0)}/4",
                    'Preview': entry.get('text_preview', '')
                })

            df = pd.DataFrame(df_data)
            csv = df.to_csv(index=False)

            st.download_button(
                "Export as CSV",
                csv,
                f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )

    with col2:
        # Export as JSON
        if filtered_history:
            json_data = json.dumps(filtered_history, indent=2)

            st.download_button(
                "Export as JSON",
                json_data,
                f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )

    with col3:
        # Clear all history
        if st.button("Clear All History", type="secondary", use_container_width=True):
            if st.session_state.get('confirm_clear', False):
                clear_history()
                st.success("History cleared")
                st.session_state.confirm_clear = False
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")
