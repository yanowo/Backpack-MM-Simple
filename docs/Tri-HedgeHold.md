### ≈ 配置提示

- `coinlist`：指定輪詢的幣種與每輪 `target_notional` / `slice_notional` 或 `slice_count`。
- `accounts`：三組 Lighter API 憑證，主帳會循環輪替，其餘兩帳即時市價對沖。
- `hold_minutes`：持倉時長（預設 17 分鐘），支持 per-symbol 覆寫。
- `entry/exit*_offset_bps`：掛單價格距離買一/賣一的偏移，避免直接衝擊市場。
- `random_split_range`：對沖拆單的隨機比例（例如 0.45~0.55）。
- 使用 `--strategy-config` 或環境變數 `VOLUME_HOLD_CONFIG` 指向實際 JSON 檔案即可啟動。