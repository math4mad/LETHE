#!/bin/bash
# 🏜️ 开放概念空间园区大门（本地网站服务）
cd website && python3 -m http.server ${1:-8000}
