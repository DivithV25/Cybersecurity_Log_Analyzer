"""
alert_engine.py
- Tracks event patterns, applies rules, and generates alerts in JSON format.
"""
from datetime import timedelta
import pandas as pd
import json

class AlertEngine:
    def __init__(self, brute_force_threshold=5, brute_force_window=2, reboot_threshold=3, reboot_window=10):
        self.brute_force_threshold = brute_force_threshold  # failed logins
        self.brute_force_window = brute_force_window        # minutes
        self.reboot_threshold = reboot_threshold            # reboots
        self.reboot_window = reboot_window                  # minutes

    def detect_brute_force(self, df: pd.DataFrame):
        alerts = []
        if 'event' not in df.columns or 'ip' not in df.columns or 'user' not in df.columns:
            return alerts
        failed = df[df['event'].str.lower() == 'failed login']
        failed = failed.dropna(subset=['ip', 'user', 'timestamp'])
        failed = failed.sort_values('timestamp')
        grouped = failed.groupby(['ip', 'user'])
        for (ip, user), group in grouped:
            times = group['timestamp'].tolist()
            for i in range(len(times) - self.brute_force_threshold + 1):
                window = times[i:i+self.brute_force_threshold]
                if (window[-1] - window[0]).total_seconds() <= self.brute_force_window * 60:
                    alerts.append({
                        "timestamp": window[-1].isoformat(),
                        "alert_type": "Brute Force",
                        "user": user,
                        "ip": ip,
                        "count": self.brute_force_threshold,
                        "severity": "High",
                        "message": f"{self.brute_force_threshold} failed logins detected from IP {ip} for user {user} — possible brute-force attack."
                    })
                    break
        return alerts

    def detect_privilege_escalation(self, df: pd.DataFrame):
        alerts = []
        if 'event' not in df.columns or 'user' not in df.columns or 'message' not in df.columns:
            return alerts
        priv = df[df['message'].str.contains('root|sudo|admin', case=False, na=False)]
        for _, row in priv.iterrows():
            if 'privilege' in row['message'] or 'escalation' in row['message']:
                alerts.append({
                    "timestamp": row['timestamp'].isoformat() if pd.notnull(row['timestamp']) else None,
                    "alert_type": "Privilege Escalation",
                    "user": row['user'],
                    "ip": row.get('ip', None),
                    "count": 1,
                    "severity": "Critical",
                    "message": f"Privilege escalation detected for user {row['user']}"
                })
        return alerts

    def detect_suspicious_access_time(self, df: pd.DataFrame, work_hours=(7, 20)):
        alerts = []
        if 'event' not in df.columns or 'timestamp' not in df.columns or 'user' not in df.columns:
            return alerts
        logins = df[df['event'].str.contains('login', case=False, na=False)]
        for _, row in logins.iterrows():
            if pd.notnull(row['timestamp']):
                hour = row['timestamp'].hour
                if hour < work_hours[0] or hour > work_hours[1]:
                    alerts.append({
                        "timestamp": row['timestamp'].isoformat(),
                        "alert_type": "Suspicious Access Time",
                        "user": row['user'],
                        "ip": row.get('ip', None),
                        "count": 1,
                        "severity": "Medium",
                        "message": f"User login at {row['timestamp'].strftime('%H:%M')} — abnormal time"
                    })
        return alerts

    def detect_frequent_reboots(self, df: pd.DataFrame):
        alerts = []
        if 'event' not in df.columns or 'timestamp' not in df.columns:
            return alerts
        reboots = df[df['event'].str.contains('reboot|restart', case=False, na=False)]
        reboots = reboots.sort_values('timestamp')
        times = reboots['timestamp'].tolist()
        for i in range(len(times) - self.reboot_threshold + 1):
            window = times[i:i+self.reboot_threshold]
            if (window[-1] - window[0]).total_seconds() <= self.reboot_window * 60:
                alerts.append({
                    "timestamp": window[-1].isoformat(),
                    "alert_type": "Frequent Reboots",
                    "user": None,
                    "ip": None,
                    "count": self.reboot_threshold,
                    "severity": "Medium",
                    "message": f"{self.reboot_threshold} system restarts within {self.reboot_window} minutes — potential instability or attack."
                })
                break
        return alerts

    def generate_alerts(self, df: pd.DataFrame) -> list:
        alerts = []
        alerts.extend(self.detect_brute_force(df))
        alerts.extend(self.detect_privilege_escalation(df))
        alerts.extend(self.detect_suspicious_access_time(df))
        alerts.extend(self.detect_frequent_reboots(df))
        return alerts
