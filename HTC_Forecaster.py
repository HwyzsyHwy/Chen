# -*- coding: utf-8 -*-
"""
HTC_Forecaster.py — 此文件内容应与 Fraud_detection-9.py 完全一致。
部署时请直接将 Fraud_detection-9.py 的内容复制到此文件。
"""
# 为方便本地测试，直接 exec 运行 Fraud_detection-9.py
import pathlib
_src = pathlib.Path(__file__).with_name("Fraud_detection-9.py")
if _src.exists():
    exec(compile(_src.read_text(encoding="utf-8"), str(_src), "exec"))
else:
    import streamlit as st
    st.error("请将 Fraud_detection-9.py 放在同目录下，或将其内容复制到本文件。")