"""
Market hours utility — determines if the market is currently open.
Handles US Eastern Time and India IST, including holidays.
"""
from __future__ import annotations

from datetime import datetime, time, date
import pytz

ET = pytz.timezone("America/New_York")
IST = pytz.timezone("Asia/Kolkata")

# US Federal holidays (approximate — update annually or use a library)
US_HOLIDAYS_2024_2025 = {
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19), date(2024, 3, 29),
    date(2024, 5, 27), date(2024, 6, 19), date(2024, 7, 4), date(2024, 9, 2),
    date(2024, 11, 28), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 1, 20), date(2025, 2, 17), date(2025, 4, 18),
    date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4), date(2025, 9, 1),
    date(2025, 11, 27), date(2025, 12, 25), date(2026, 1, 1),
}

# NSE India holidays 2025-2026 (official NSE trading holidays)
NSE_HOLIDAYS_2025_2026 = {
    # 2025
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan Eid)
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 5, 12),   # Buddha Purnima
    date(2025, 6, 7),    # Bakri Id (Eid al-Adha)
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Ashura
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali (Lakshmi Puja)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti / Prakash Gurpurab
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri
    date(2026, 3, 4),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr (Ramadan Eid)
    date(2026, 3, 30),   # Shri Mahavir Jayanti
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day / Buddha Purnima
    date(2026, 5, 27),   # Bakri Id (Eid al-Adha)
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 18),   # Janmashtami
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 20),  # Dussehra
    date(2026, 11, 9),   # Diwali (Lakshmi Puja)
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}


# Muhurat Trading sessions (Diwali — special Sunday trading)
MUHURAT_SESSIONS = {
    date(2025, 10, 21): (time(18, 15), time(19, 15)),  # Diwali 2025 (approx 1hr evening)
    date(2026, 11, 9): (time(18, 15), time(19, 15)),   # Diwali 2026 (approx 1hr evening)
}

# F&O expiry shifts: if Thursday is a holiday, expiry moves to Wednesday
# Format: original_date -> shifted_date
FNO_EXPIRY_SHIFTS_2025_2026: dict[date, date] = {
    # Add known shifts here as they're announced by NSE
}


class MarketHours:

    def __init__(self, exchange: str = "US"):
        self.exchange = exchange
        self._tz = ET if exchange == "US" else IST

    def is_open(self) -> bool:
        """Return True if the market is currently in core trading hours."""
        now = datetime.now(self._tz)

        # Check for Muhurat trading session (special Sunday Diwali session)
        if self.exchange == "IN" and now.date() in MUHURAT_SESSIONS:
            muhurat_open, muhurat_close = MUHURAT_SESSIONS[now.date()]
            if muhurat_open <= now.time() <= muhurat_close:
                return True

        return self._is_trading_day(now) and self._is_trading_time(now)

    def is_pre_market(self) -> bool:
        """US pre-market: 4:00 AM – 9:30 AM ET."""
        if self.exchange != "US":
            return False
        now = datetime.now(ET)
        if not self._is_trading_day(now):
            return False
        t = now.time()
        return time(4, 0) <= t < time(9, 30)

    def is_after_hours(self) -> bool:
        """US after-hours: 4:00 PM – 8:00 PM ET."""
        if self.exchange != "US":
            return False
        now = datetime.now(ET)
        if not self._is_trading_day(now):
            return False
        t = now.time()
        return time(16, 0) <= t <= time(20, 0)

    def minutes_to_open(self) -> int:
        """Return minutes until next market open. 0 if market is open."""
        if self.is_open():
            return 0
        now = datetime.now(self._tz)
        open_time = time(9, 35) if self.exchange == "US" else time(9, 20)
        today_open = now.replace(
            hour=open_time.hour, minute=open_time.minute,
            second=0, microsecond=0
        )
        if now.time() > open_time:
            # Already past open today — next open is tomorrow
            from datetime import timedelta
            today_open += timedelta(days=1)
        diff = today_open - now
        return max(0, int(diff.total_seconds() / 60))

    def next_market_open_str(self) -> str:
        """Human-readable next market open."""
        now = datetime.now(self._tz)
        if self.is_open():
            return "NOW"
        minutes = self.minutes_to_open()
        if minutes < 60:
            return f"in {minutes}m"
        return f"in {minutes // 60}h {minutes % 60}m"

    def _is_trading_day(self, dt: datetime) -> bool:
        if dt.weekday() >= 5:  # Weekend
            return False
        if self.exchange == "US" and dt.date() in US_HOLIDAYS_2024_2025:
            return False
        if self.exchange == "IN" and dt.date() in NSE_HOLIDAYS_2025_2026:
            return False
        return True

    def _is_trading_time(self, dt: datetime) -> bool:
        t = dt.time()
        if self.exchange == "US":
            return time(9, 35) <= t <= time(15, 45)
        else:  # India
            return time(9, 20) <= t <= time(15, 20)
