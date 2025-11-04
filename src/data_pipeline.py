import re
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel


LOG_PATTERN = re.compile(
    r'^(?P<month>\w{3}) (?P<day>\d{1,2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<host>\S+) (?P<process>\S+): (?P<message>.*)$'
)
MONTHS = {m: i for i, m in enumerate(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], 1)}


class LogDataPipeline:
    """Small, self-contained log pipeline used by the sample app.

    It provides a clean, easy-to-understand implementation so the app can run
    while you iterate on features.
    """

    def __init__(self, embedding_model: str = 'distilbert-base-uncased'):
        # load small embedding model (Hugging Face)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.model.eval()

    def parse_log_line(self, line: str, year: Optional[int] = None) -> Optional[Dict]:
        m = LOG_PATTERN.match(line)
        if not m:
            return None
        d = m.groupdict()
        try:
            month = MONTHS.get(d['month'], 1)
            day = int(d['day'])
            time_str = d['time']
            year = year or datetime.now().year
            timestamp = datetime.strptime(f"{year}-{month:02d}-{day:02d} {time_str}", "%Y-%m-%d %H:%M:%S")
        except Exception:
            timestamp = None
        return {
            'timestamp': timestamp,
            'host': d.get('host'),
            'process': d.get('process'),
            'message': d.get('message'),
            'raw': line,
        }

    def clean_message(self, msg: str) -> str:
        return re.sub(r'\s+', ' ', (msg or '')).strip()

    def extract_features(self, entry: Dict) -> Dict:
        msg = (entry.get('message') or '')
        ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', msg)
        # sudo line pattern: "sudo:    alice : TTY=pts/1 ; PWD=/home/alice ; USER=root ; COMMAND=/bin/su -"
        sudo_re = re.compile(r"sudo:\s*(?P<performer>[\w-]+)\s*:\s*.*USER=(?P<target>[\w-]+)\s*;\s*COMMAND=(?P<cmd>.+)", flags=re.IGNORECASE)
        su_re = re.compile(r"su\[\d+\]: (?P<msg>.+)")
        user_match = re.search(r'user ([\w-]+)', msg, flags=re.IGNORECASE)
        accepted_re = re.compile(r'Accepted password for (?P<user>[\w-]+)', flags=re.IGNORECASE)
        msg_l = msg.lower()
        # classify event: failed/accepted login, reboot, or other
        if 'failed password' in msg_l or 'failed login' in msg_l:
            event = 'failed login'
        elif 'accepted password' in msg_l or ('accepted' in msg_l and 'password' in msg_l):
            event = 'accepted login'
        elif 'booting' in msg_l or 'startup finished' in msg_l or ('boot' in msg_l and 'linux' in msg_l) or 'reboot' in msg_l or 'restart' in msg_l:
            event = 'reboot'
        else:
            event = 'other'

        performer = None
        target_user = None
        command = None
        m = sudo_re.search(msg)
        if m:
            performer = m.group('performer')
            target_user = m.group('target')
            command = m.group('cmd').strip()
        else:
            # try to parse typical su messages
            m2 = re.search(r'for user (?P<target>[\w-]+)', msg, flags=re.IGNORECASE)
            if m2:
                target_user = m2.group('target')

        # if performer absent, use user_match as performer
        if performer is None:
            # prefer explicit Accepted login user
            m_acc = accepted_re.search(msg)
            if m_acc:
                performer = m_acc.group('user')
                event = 'accepted login'
            elif user_match:
                performer = user_match.group(1)

        return {
            **entry,
            'ip': ip_match.group(1) if ip_match else None,
            'user': performer,
            'target_user': target_user,
            'command': command,
            'event': event,
        }

    def log_to_embedding(self, text: str) -> np.ndarray:
        # return a numpy vector for the input text; small and deterministic
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64)
        with torch.no_grad():
            outputs = self.model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return emb

    def process_log_file(self, filepath: str, year: Optional[int] = None) -> pd.DataFrame:
        rows = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parsed = self.parse_log_line(line.strip(), year=year)
                if parsed is None:
                    continue
                parsed['message'] = self.clean_message(parsed.get('message', ''))
                features = self.extract_features(parsed)
                try:
                    features['embedding'] = self.log_to_embedding(features['message'])
                except Exception:
                    # embedding generation should not block basic parsing
                    features['embedding'] = None
                rows.append(features)
        df = pd.DataFrame(rows)
        return df

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna('')

    def process(self, filepath: str) -> pd.DataFrame:
        df = self.process_log_file(filepath)
        return self.clean_df(df)

