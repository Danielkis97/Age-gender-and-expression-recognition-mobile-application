# Mobile Browser Test (Windows + iPhone)

This setup lets the iPhone run inference in Safari and sends latency samples to the Windows PC for direct evaluation.

## 1) Start local collector server on Windows

From project root:

```powershell
.\.venv\Scripts\python.exe -u mobile_eval_server.py --host 0.0.0.0 --port 8000 --model_path models/model.tflite --images_dir dataset/images --out_csv "results/Results mobile metrics/mobile_browser_metrics.csv"
```

The server prints two URLs:

- PC test page: `http://127.0.0.1:8000/mobile`
- iPhone page: `http://<your-pc-lan-ip>:8000/mobile`

Find your PC LAN IP in PowerShell:

```powershell
(Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null } | Select-Object -First 1).IPv4Address.IPAddress
```

If PowerShell shows `>>`, the command was not parsed as complete.
Press `Ctrl+C`, then paste and run the full command again.

## 2) Open page on iPhone (same WLAN)

Open Safari and enter:

```text
http://<your-pc-lan-ip>:8000/mobile
```

Default behavior:

- Input source is `dataset/images` automatically (first 20 images)
- Warm-up runs are applied to the **first image only** (`2` warm-up calls)
- Measured pass is **exactly one run** over all 20 images (`n_runs = 1`)

If you still see old dropdowns/options in Safari, force-reload once (or close/reopen the tab).

## 3) Collect metrics

1. Tap **Load TFLite Model**
2. Tap **Run Benchmark and Upload Metrics**
3. Samples are written to:

`results/Results mobile metrics/mobile_browser_metrics.csv`

4. Optional KPI summary on PC:

```powershell
.\.venv\Scripts\python.exe results/summarize_mobile_metrics.py --csv "results/Results mobile metrics/mobile_browser_metrics.csv"
```

## 4) Quick sanity check

Tap **Send One Test Row** to verify that iPhone can reach the PC collector.

## Notes

- This is browser-based on-device inference (not a native iOS app).
- iPhone and PC must be in the same local network.
- Allow Python/port 8000 through Windows Firewall for private networks.
