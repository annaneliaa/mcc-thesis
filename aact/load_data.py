import json
import pandas as pd
from glob import glob
import os

def extract_scenario(filename: str) -> str:
    base = os.path.basename(filename)
    name, _ = os.path.splitext(base)
    return name.split("_", 1)[0] if "_" in name else name


def load_alerts_from_json(output_file, dir_path):
    files = glob(f"{dir_path}/*.json")
    rows = []

    for file in files:
        print(f"Opening file {file}...")
        with open(file, "r") as f:
            for line in f:
                scenario = extract_scenario(file)
                obj = json.loads(line)

                # ---------- AMiner ----------
                if "AnalysisComponent" in obj:
                    ts = obj.get("LogData", {}).get("DetectionTimestamp", [None])[0]
                    category = obj.get("AnalysisComponent", {}).get(
                        "AnalysisComponentName", "UNKNOWN"
                    )
                    entity = obj.get("AMiner", {}).get("ID", "UNKNOWN")
                    raw_log = obj.get("LogData", {}).get("RawLogData", [""])[0]
                    source = "aminer"

                    aminer_component_type = obj.get("AnalysisComponent", {}).get(
                        "AnalysisComponentType", "UNKNOWN"
                    )
                    aminer_training_mode = int(
                        obj.get("AnalysisComponent", {}).get("TrainingMode", False)
                    )
                    aminer_new_event = int("new event" in category.lower())

                    wazuh_level = None
                    wazuh_antivirus = 0
                    wazuh_update = 0

                # ---------- Wazuh ----------
                elif "@timestamp" in obj:
                    ts = obj.get("@timestamp")
                    category = obj.get("rule", {}).get("description", "UNKNOWN")
                    entity = (
                        obj.get("agent", {}).get("ip")
                        or obj.get("predecoder", {}).get("hostname")
                        or "UNKNOWN"
                    )
                    raw_log = obj.get("full_log", "")
                    source = "wazuh"

                    wazuh_level = obj.get("rule", {}).get("level")
                    wazuh_antivirus = int(
                        any(
                            g in ["clamd", "freshclam", "virus"]
                            for g in obj.get("rule", {}).get("groups", [])
                        )
                    )
                    wazuh_update = int("update" in category.lower())

                    aminer_component_type = None
                    aminer_training_mode = 0
                    aminer_new_event = 0

                else:
                    continue

                raw_lower = raw_log.lower()

                rows.append({
                    "timestamp": ts,
                    "category": category,
                    "entity": entity,
                    "raw_log": raw_log,
                    "scenario": scenario,
                    "source": source,

                    # --- static semantic features ---
                    "is_auth_event": int(
                        any(k in raw_lower for k in ["auth", "login", "pam"])
                    ),
                    "is_cred_event": int("cred" in raw_lower),
                    "is_web_event": int(
                        any(k in raw_lower for k in ["http", "wp", "apache", "nginx"])
                    ),
                    "is_cron": int("cron" in raw_lower),
                    "is_success": int("res=success" in raw_lower),
                    "is_uid0": int("uid=0" in raw_lower),

                    # --- AMiner specific ---
                    "aminer_component_type": aminer_component_type,
                    "aminer_training_mode": aminer_training_mode,
                    "aminer_new_event": aminer_new_event,

                    # --- Wazuh specific ---
                    "wazuh_level": wazuh_level,
                    "wazuh_antivirus": wazuh_antivirus,
                    "wazuh_update": wazuh_update,
                })

    df = pd.DataFrame(rows)

    # --- normalize timestamps ---
    mask_aminer = df["source"] == "aminer"
    df.loc[mask_aminer, "timestamp"] = pd.to_datetime(
        df.loc[mask_aminer, "timestamp"], unit="s", utc=True, errors="coerce"
    )

    mask_wazuh = df["source"] == "wazuh"
    df.loc[mask_wazuh, "timestamp"] = pd.to_datetime(
        df.loc[mask_wazuh, "timestamp"], utc=True, format="mixed", errors="coerce"
    )

    df = df.dropna(subset=["timestamp", "category", "entity"])

    print(f"Writing data to {output_file}...")
    df.to_csv(f"../data/ait_ads/{output_file}", index=False)
    print("Done.")
