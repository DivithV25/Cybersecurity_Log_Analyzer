"""nlp_interface_enhanced.py
- Enhanced natural language query interface for logs and alerts.
- Provides context-aware cybersecurity analysis via LLMs.
"""
import pandas as pd
import logging
from typing import Optional

try:
    from langchain.prompts import PromptTemplate
    from transformers import pipeline
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class NLPLLMInterface:
    """Natural language query interface for cybersecurity log analysis."""
    
    def __init__(self, model_name: str = 't5-small'):
        """Initialize NLP interface with specified model.
        
        Args:
            model_name: HF model for text2text-generation (default: t5-small for speed).
        """
        self.model_name = model_name
        self.enabled = LANGCHAIN_AVAILABLE
        self.qa_pipeline = None
        
        if self.enabled:
            try:
                # Use lightweight model (t5-small) for faster inference
                self.qa_pipeline = pipeline('text2text-generation', model=model_name, device=-1)
                logger.info(f"NLP interface initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize NLP interface: {e}")
                self.enabled = False
        
        self.prompt_template = """Analyze security logs and answer the question.

LOGS:
{logs}

ALERTS:
{alerts}

Q: {query}
A:"""

    def query(self, query: str, logs: pd.DataFrame, alerts: list) -> str:
        """Query logs and alerts using natural language.
        
        Args:
            query: Natural language question.
            logs: Parsed logs DataFrame.
            alerts: List of generated alerts.
            
        Returns:
            Natural language response or error message.
        """
        if not self.enabled or self.qa_pipeline is None:
            return "NLP interface is not available. Please ensure transformers is installed."
        
        try:
            # Extract unique IPs and users for context
            unique_ips = logs['ip'].dropna().unique().tolist() if 'ip' in logs.columns else []
            unique_users = logs['user'].dropna().unique().tolist() if 'user' in logs.columns else []
            
            # Check if this is a query that can be answered with direct extraction
            is_basic_query = self._is_basic_query(query)
            
            if is_basic_query:
                direct_response = self._extract_direct_answer(query, logs, alerts, unique_ips, unique_users)
                if direct_response:
                    return direct_response
            
            # For complex queries, prepare data and use T5 model
            # Prepare logs - focus on relevant columns only
            cols_to_use = [c for c in ['timestamp', 'event', 'user', 'ip', 'message'] if c in logs.columns]
            logs_str = logs[cols_to_use].head(25).to_string(index=False) if len(logs) > 0 else "No logs"
            
            # Prepare alerts - concise format
            if alerts:
                alerts_str = '\n'.join([
                    f"[{a.get('alert_type', 'Unknown')}] {a.get('message', 'N/A')}"
                    for a in alerts[:5]
                ])
            else:
                alerts_str = "No alerts"
            
            # Format prompt - KEEP IT SHORT to avoid echo-back
            full_prompt = self.prompt_template.format(
                query=query,
                logs=logs_str,
                alerts=alerts_str,
                ips=', '.join(unique_ips) if unique_ips else 'None',
                users=', '.join(unique_users) if unique_users else 'None'
            )
            
            # Run T5 inference
            result = self.qa_pipeline(full_prompt, max_length=150, do_sample=False, temperature=0.05)
            full_response = result[0]['generated_text']
            
            # Extract ONLY the answer part (after "A:")
            if '\nA:' in full_response:
                answer = full_response.split('\nA:')[-1].strip()
            else:
                answer = full_response.replace(full_prompt, '').strip()
            
            # Clean up response - remove any remaining prompt elements or raw data
            answer = self._clean_response(answer, logs_str, alerts_str)
            
            # If answer is empty, too short, or contains prompt elements, use direct extraction fallback
            if not answer or len(answer) < 3 or 'Q:' in answer or 'LOGS:' in answer or 'ALERTS:' in answer or '2025-' in answer:
                direct = self._extract_direct_answer(query, logs, alerts, unique_ips, unique_users)
                return direct or f"Unable to find answer for: {query}"
            
            return answer
        except Exception as e:
            logger.error(f"Error during NLP query: {e}")
            return f"Error: {str(e)}"
    
    def _clean_response(self, response: str, logs_context: str, alerts_context: str) -> str:
        """Clean response by removing raw log data and formatting artifacts."""
        # Remove raw timestamps and log entries that got echoed
        import re
        
        # Remove date patterns (YYYY-MM-DD)
        response = re.sub(r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\w+\s+login\s+\w+\s+[\d.]+\s+', '', response)
        
        # Remove lines that look like raw logs
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip lines that are clearly raw log entries
            if re.match(r'\d{4}-\d{2}-\d{2}', line):  # Starts with timestamp
                continue
            if 'Failed password' in line or 'failed login' in line:  # Raw log content
                continue
            if 'ALERTS:' in line or 'LOGS:' in line or 'Q:' in line:  # Prompt remnants
                continue
            cleaned_lines.append(line)
        
        response = '\n'.join(cleaned_lines).strip()
        
        # Ensure response is concise
        if response.startswith('['):
            response = response.strip('[]').strip()
        
        return response
    
    def _is_basic_query(self, query: str) -> bool:
        """Determine if query is basic enough for direct extraction."""
        query_lower = query.lower()
        
        # Basic queries that don't need T5 - factual data extraction
        basic_keywords = [
            # IPs
            'which ip', 'what ip', 'list ip', 'ip address',
            # Users
            'which user', 'what user', 'list user', 'list all user', 'all users', 'who access',
            # Ports
            'which port', 'what port', 'list port', 'port',
            # Alerts
            'how many alert', 'alert count', 'total alert', 'number of alert',
            # Severity
            'severity', 'critical', 'high severity',
            # Attack-related countable queries (MUST be before complex indicators)
            'how many', 'how many times', 'how often', 'count of', 'number of',
            'how many brute', 'brute force attack occur', 'brute force attack count',
            'how many escalation', 'escalation attack count', 'privilege escalation attack',
            'how many suspicious', 'suspicious activity', 'how many attack'
        ]
        
        # Check if it matches basic keywords
        matches_basic = any(keyword in query_lower for keyword in basic_keywords)
        if not matches_basic:
            return False
        
        # Check for complex query indicators that should override basic routing
        # (only if they indicate actual analysis, not just asking about what an attack is)
        complex_indicators = [
            'describe', 'explain', 'analyze', 'detail', 'timeline', 'sequence',
            'what happened', 'why', 'how did', 'pattern', 'summarize'
        ]
        
        # If matches basic AND has complex indicator, check context
        for indicator in complex_indicators:
            if indicator in query_lower:
                # "describe the attack" = complex, but "what is X attack?" = basic
                if not any(attack_type in query_lower for attack_type in 
                          ['brute force', 'escalation', 'privilege', 'suspicious']):
                    return False
        
        return matches_basic
    
    def _extract_direct_answer(self, query: str, logs: pd.DataFrame, alerts: list, unique_ips: list, unique_users: list) -> Optional[str]:
        """Extract answers directly from data - handles specific queries efficiently."""
        import re
        query_lower = query.lower()
        
        # Attack occurrence queries - count brute force, escalation, etc.
        if any(phrase in query_lower for phrase in ['how many times', 'how many brute', 'brute force attack occur', 'how often', 'attack count']):
            brute_force_alerts = [a for a in alerts if 'brute force' in a.get('alert_type', '').lower()]
            if brute_force_alerts:
                # Extract attempt count from alert message
                total_attempts = 0
                for alert in brute_force_alerts:
                    msg = alert.get('message', '')
                    # Try to extract number of failed logins from message
                    match = re.search(r'(\d+)\s+failed\s+login', msg, re.IGNORECASE)
                    if match:
                        total_attempts += int(match.group(1))
                    # Also try alternate patterns
                    match = re.search(r'(\d+)\s+attempt', msg, re.IGNORECASE)
                    if match and not (match.start() > 0 and msg[match.start()-1].isdigit()):
                        total_attempts += int(match.group(1))
                
                # If we found attempts in message, report that count
                if total_attempts > 0:
                    return f"Brute force attack: {total_attempts} failed login attempts detected. Details: {brute_force_alerts[0].get('message', 'N/A')}"
                # Otherwise count number of brute force alerts
                else:
                    return f"Brute force attack occurred {len(brute_force_alerts)} time(s). Details: {brute_force_alerts[0].get('message', 'N/A')}"
            return "No brute force attacks detected in the logs."
        
        # Privilege escalation queries
        if any(phrase in query_lower for phrase in ['privilege escalation', 'how many escalation', 'escalation attack']):
            escalation_alerts = [a for a in alerts if 'privilege' in a.get('alert_type', '').lower() or 'escalation' in a.get('alert_type', '').lower()]
            if escalation_alerts:
                # Extract attempt count from message
                total_attempts = 0
                for alert in escalation_alerts:
                    msg = alert.get('message', '')
                    match = re.search(r'(\d+)\s+attempt', msg, re.IGNORECASE)
                    if match:
                        total_attempts += int(match.group(1))
                
                if total_attempts > 0:
                    return f"Privilege escalation: {total_attempts} attempts detected. Details: {escalation_alerts[0].get('message', 'N/A')}"
                else:
                    return f"Privilege escalation detected {len(escalation_alerts)} time(s). Details: {escalation_alerts[0].get('message', 'N/A')}"
            return "No privilege escalation attempts detected."
        
        # Suspicious activity queries
        if any(phrase in query_lower for phrase in ['suspicious', 'how many suspicious', 'suspicious activity']):
            suspicious_alerts = [a for a in alerts if any(word in a.get('alert_type', '').lower() for word in ['suspicious', 'anomaly', 'unusual'])]
            if suspicious_alerts:
                return f"Suspicious activities detected: {len(suspicious_alerts)} event(s)."
            return "No suspicious activities detected."
        
        # Generic attack count queries
        if any(phrase in query_lower for phrase in ['how many attack', 'total attack', 'attack count', 'number of attack']):
            return f"Total attacks/security events detected: {len(alerts)}."
        
        # Port queries - search logs for port numbers
        if any(word in query_lower for word in ['port', 'ports', 'attacked']):
            ports_found = self._extract_ports_from_logs(logs)
            if ports_found:
                return f"Ports detected in logs: {', '.join(ports_found)}."
            return "No specific port information found in logs."
        
        # IPs queries
        if any(phrase in query_lower for phrase in ['which ip', 'what ip', 'list ip', 'ips in', 'ip address']):
            if unique_ips:
                return f"IPs found in logs: {', '.join(sorted(unique_ips))}."
            return "No IPs found in logs."
        
        # Users queries
        if any(phrase in query_lower for phrase in ['which user', 'what user', 'list user', 'users in', 'all users']):
            if unique_users:
                return f"Users found in logs: {', '.join(sorted(unique_users))}."
            return "No users found in logs."
        
        # Alert count queries
        if any(kw in query_lower for kw in ['how many alert', 'alert count', 'total alert', 'number of alert']):
            return f"Total alerts detected: {len(alerts)}."
        
        # Severity queries
        if any(kw in query_lower for kw in ['severity', 'critical', 'high severity']):
            if alerts:
                severity_count = {}
                for alert in alerts:
                    sev = alert.get('severity', 'Unknown')
                    severity_count[sev] = severity_count.get(sev, 0) + 1
                breakdown = ', '.join([f'{k}: {v}' for k, v in sorted(severity_count.items())])
                return f"Alert severity breakdown: {breakdown}."
            return "No alerts to analyze."
        
        return None
    
    def _extract_ports_from_logs(self, logs: pd.DataFrame) -> list:
        """Extract port numbers from log messages using regex."""
        import re
        ports = set()
        if 'message' not in logs.columns:
            return []
        
        # Patterns to match ports
        patterns = [
            r'port\s+(\d+)',           # "port 22"
            r':(\d+)\s+ssh',           # ":22 ssh"
            r':\d+\s+(\d+)',           # ":53311 54321"
            r'port[=\s]+(\d+)',        # "port=3306"
        ]
        
        for _, row in logs.iterrows():
            msg = str(row.get('message', '')).lower()
            for pattern in patterns:
                matches = re.findall(pattern, msg)
                for match in matches:
                    port_num = int(match)
                    if 1 <= port_num <= 65535:  # Valid port range
                        ports.add(str(port_num))
        
        return sorted(list(ports))
