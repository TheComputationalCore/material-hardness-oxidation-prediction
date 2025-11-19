"""
utils.py — helper utilities for the Flask app.
"""

def safe_float(value):
    try:
        return float(value)
    except:
        return None
