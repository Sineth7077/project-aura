"""Optional Streamlit analytics dashboard for Project Aura.

Run with:
    streamlit run src/dashboard/app.py

This dashboard displays real data from the monitoring system when available.
It is OPTIONAL — the notebooks and API work without Streamlit.
"""

from __future__ import annotations

import json
import os


def _load_metrics_json() -> dict | None:
    """Attempt to load metrics from a previous notebook export."""
    for path in ["output/metrics.json", "/content/output/metrics.json"]:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def main() -> None:
    """Run the Streamlit analytics dashboard."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install with: pip install streamlit")
        return

    st.set_page_config(page_title="Project Aura Analytics", layout="wide")
    st.title("Project Aura 2.0 — Analytics Dashboard")

    metrics = _load_metrics_json()
    if metrics is None:
        st.info("No metrics data found. Run Notebook 08 to generate metrics exports in `output/metrics.json`, or use the live resource monitor below.")

    # Live resource monitor (always available)
    with st.expander("Live System Resources"):
        try:
            from src.monitoring.resource_monitor import ResourceMonitor
            monitor = ResourceMonitor(check_gpu=True)
            snap = monitor.get_snapshot().to_dict()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CPU %", f"{snap['cpu_percent']:.1f}" if snap["cpu_percent"] is not None else "N/A")
                st.metric("RAM %", f"{snap['ram_percent']:.1f}" if snap["ram_percent"] is not None else "N/A")
            with col2:
                st.metric("Process Memory (MB)", f"{snap['process_memory_mb']:.1f}" if snap["process_memory_mb"] else "N/A")
                st.metric("GPU Available", str(snap["gpu_available"]))
        except Exception as exc:
            st.error(f"Resource monitoring unavailable: {exc}")

    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Select view",
        ["Overview", "Inference Metrics", "Gesture Analytics", "Agent Analytics", "Export"],
    )

    if page == "Overview":
        st.header("System Overview")
        if metrics:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Inference Requests", metrics.get("inference", {}).get("total_requests", "N/A"))
            with c2:
                st.metric("Gestures Detected", metrics.get("gesture", {}).get("total_gestures", "N/A"))
            with c3:
                st.metric("Agent Observations", metrics.get("agent", {}).get("total_observations", "N/A"))
            with c4:
                st.metric("Avg Latency (ms)", f"{metrics.get('inference', {}).get('avg_latency_ms', 0):.1f}")
        else:
            st.warning("No metrics loaded. Run Notebook 08 first.")

    elif page == "Inference Metrics":
        st.header("Inference Metrics")
        if metrics and "inference" in metrics:
            inf = metrics["inference"]
            st.json(inf)
        else:
            st.info("No inference metrics loaded.")

    elif page == "Gesture Analytics":
        st.header("Gesture Analytics")
        if metrics and "gesture" in metrics:
            gest = metrics["gesture"]
            st.json(gest)
            if "distribution" in gest:
                st.bar_chart(gest["distribution"])
        else:
            st.info("No gesture metrics loaded.")

    elif page == "Agent Analytics":
        st.header("Agent Analytics")
        if metrics and "agent" in metrics:
            ag = metrics["agent"]
            st.json(ag)
            if "decisions_by_type" in ag:
                st.bar_chart(ag["decisions_by_type"])
        else:
            st.info("No agent metrics loaded.")

    elif page == "Export":
        st.header("Export Data")
        if metrics:
            st.download_button(
                "Download metrics JSON",
                json.dumps(metrics, indent=2, default=str),
                "aura_metrics.json",
                "application/json",
            )
        else:
            st.info("No data to export.")


if __name__ == "__main__":
    main()
