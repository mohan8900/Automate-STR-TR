"""
Alert manager — multi-channel notifications with priority tiers.
CRITICAL → SMS + Email
HIGH     → Email
MEDIUM   → Slack webhook
LOW      → Dashboard only (logged)
"""
from __future__ import annotations

import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from typing import Optional

import requests

from config.settings import TradingSystemConfig
from core.logger import get_logger

log = get_logger("alerts")


class AlertManager:

    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.notif = config.notifications

    # ── Priority tiers ────────────────────────────────────────────────────

    def send_critical(self, message: str) -> None:
        """CRITICAL: Circuit breaker, system errors. SMS + Email."""
        log.warning(f"[CRITICAL ALERT] {message}")
        if self.notif.alert_on_circuit_break:
            self._send_email(f"[CRITICAL] AI Trader Alert", message, priority="high")
            self._send_sms(f"CRITICAL ALERT: {message[:140]}")

    def send_high(self, message: str) -> None:
        """HIGH: Trade executions, stop-loss hits, approval needed. Email."""
        log.info(f"[HIGH ALERT] {message}")
        if self.notif.alert_on_trade:
            self._send_email("[ACTION] AI Trader", message)
            self._send_slack(f"⚠️ {message}")

    def send_signal(self, message: str) -> None:
        """MEDIUM: New signal generated."""
        log.info(f"[SIGNAL] {message}")
        self._send_slack(f"📊 {message}")

    def send_trade_executed(self, trade) -> None:
        """HIGH: Trade was executed."""
        msg = (
            f"✅ TRADE EXECUTED\n"
            f"Symbol: {trade.symbol}\n"
            f"Action: {trade.action}\n"
            f"Shares: {trade.shares:.0f}\n"
            f"Entry: ${trade.entry_price:.2f}\n"
            f"Value: ${trade.position_value:,.0f}\n"
            f"Stop Loss: ${trade.stop_loss_price:.2f}\n"
            f"Conviction: {trade.conviction}/10\n"
            f"Thesis: {trade.primary_thesis[:100]}"
        )
        log.info(f"[TRADE EXECUTED] {trade.symbol} {trade.action}")
        if self.notif.alert_on_trade:
            self._send_email(f"Trade Executed: {trade.action} {trade.symbol}", msg)
            self._send_slack(f"✅ Trade executed: {trade.action} {trade.shares:.0f} {trade.symbol} @ ~${trade.entry_price:.2f}")

    def send_daily_summary(self, summary: str) -> None:
        """Daily performance summary. Email + Slack."""
        log.info("[DAILY SUMMARY] Sending...")
        self._send_email("AI Trader Daily Summary", summary)
        self._send_slack(f"📈 Daily Summary\n{summary}")

    # ── Delivery channels ─────────────────────────────────────────────────

    def _send_email(self, subject: str, body: str, priority: str = "normal") -> None:
        if not self.notif.email_enabled:
            return
        try:
            import os
            smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USER", "")
            smtp_pass = os.getenv("SMTP_PASS", "")
            alert_email = os.getenv("ALERT_EMAIL", smtp_user)

            if not smtp_user or not smtp_pass:
                return

            msg = MIMEText(f"{body}\n\n—\nAI Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = alert_email
            if priority == "high":
                msg["X-Priority"] = "1"

            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
            log.debug(f"Email sent: {subject}")
        except Exception as e:
            log.debug(f"Email send failed: {e}")

    def _send_sms(self, message: str) -> None:
        if not self.notif.sms_enabled:
            return
        try:
            import os
            from twilio.rest import Client
            sid = os.getenv("TWILIO_ACCOUNT_SID", "")
            token = os.getenv("TWILIO_AUTH_TOKEN", "")
            from_num = os.getenv("TWILIO_FROM_NUMBER", "")
            to_num = os.getenv("ALERT_PHONE_NUMBER", "")
            if not all([sid, token, from_num, to_num]):
                return
            client = Client(sid, token)
            client.messages.create(body=message, from_=from_num, to=to_num)
            log.debug("SMS sent")
        except Exception as e:
            log.debug(f"SMS send failed: {e}")

    def _send_slack(self, message: str) -> None:
        if not self.notif.slack_enabled:
            return
        try:
            import os
            webhook = os.getenv("SLACK_WEBHOOK_URL", "")
            if not webhook:
                return
            requests.post(webhook, json={"text": message}, timeout=5)
            log.debug("Slack message sent")
        except Exception as e:
            log.debug(f"Slack send failed: {e}")
