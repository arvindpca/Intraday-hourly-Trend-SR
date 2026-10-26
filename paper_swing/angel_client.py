"""
angel_client.py
----------------
Handles login to Angel One SmartAPI using API key + client id + PIN + TOTP.

The module:
- Loads credentials from .env
- Generates the current TOTP
- Logs into Angel One SmartAPI
- Retries login with a fresh TOTP if AB1050 occurs
- Keeps the active SmartConnect object in memory
"""

import os
import time
import unicodedata

import pyotp
from SmartApi import SmartConnect
from dotenv import load_dotenv


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


def _clean_credential(value):
    """Remove invisible Unicode format marks accidentally copied into secrets.

    SmartAPI sends API key/client ID in HTTP headers; characters such as U+200E
    (LEFT-TO-RIGHT MARK) cannot be encoded in Latin-1 HTTP headers and crash
    the request before it reaches Angel One. Do not log the credential values.
    """
    if value is None:
        return None
    return "".join(ch for ch in value.strip() if unicodedata.category(ch) != "Cf")


API_KEY = _clean_credential(os.getenv("ANGEL_API_KEY"))
CLIENT_ID = _clean_credential(os.getenv("ANGEL_CLIENT_ID"))
PASSWORD = _clean_credential(os.getenv("ANGEL_PASSWORD"))
TOTP_SECRET = _clean_credential(os.getenv("ANGEL_TOTP_SECRET"))


# ---------------------------------------------------------
# Session variables
# ---------------------------------------------------------

_smart_api = None
_session_data = None


# ---------------------------------------------------------
# TOTP
# ---------------------------------------------------------

def get_totp_code() -> str:
    """
    Generate the current 6-digit TOTP code.

    The code is generated from the same TOTP secret
    configured in the Angel One Authenticator setup.
    """

    if not TOTP_SECRET:
        raise RuntimeError("ANGEL_TOTP_SECRET is missing in .env")

    totp = pyotp.TOTP(TOTP_SECRET)

    return totp.now()


# ---------------------------------------------------------
# Login
# ---------------------------------------------------------

def login():
    """
    Login to Angel One SmartAPI.

    If login fails, a fresh TOTP is generated and the
    login is retried up to 3 times.
    """

    global _smart_api
    global _session_data

    # -----------------------------------------------------
    # Check credentials
    # -----------------------------------------------------

    if not all([
        API_KEY,
        CLIENT_ID,
        PASSWORD,
        TOTP_SECRET
    ]):
        raise RuntimeError(
            "Missing Angel One credentials. "
            "Please check your .env file."
        )

    # -----------------------------------------------------
    # Maximum login attempts
    # -----------------------------------------------------

    max_attempts = 3

    # -----------------------------------------------------
    # Login attempts
    # -----------------------------------------------------

    for attempt in range(1, max_attempts + 1):

        print(
            f"[angel_client] Login attempt "
            f"{attempt}/{max_attempts} (TOTP generated)"
        )

        # -------------------------------------------------
        # Create fresh SmartConnect object
        # -------------------------------------------------

        smart_api = SmartConnect(
            api_key=API_KEY
        )

        # -------------------------------------------------
        # Generate fresh TOTP
        # -------------------------------------------------

        totp_code = get_totp_code()

        # -------------------------------------------------
        # Login
        # -------------------------------------------------

        session_data = smart_api.generateSession(
            CLIENT_ID,
            PASSWORD,
            totp_code
        )

        # -------------------------------------------------
        # Login successful
        # -------------------------------------------------

        if session_data.get("status"):

            _smart_api = smart_api
            _session_data = session_data

            print(
                "[angel_client] Logged in to Angel One "
                "SmartAPI successfully."
            )

            return smart_api

        # -------------------------------------------------
        # Login failed
        # -------------------------------------------------

        message = session_data.get(
            "message",
            "Unknown login error"
        )

        error_code = session_data.get(
            "errorcode",
            ""
        )

        print(
            f"[angel_client] Login failed: "
            f"{message} {error_code}"
        )

        # -------------------------------------------------
        # Wait for next TOTP before retry
        # -------------------------------------------------

        if attempt < max_attempts:

            # TOTP normally changes every 30 seconds.
            # Wait until the current TOTP window is nearly
            # finished, then generate a new one.

            remaining = 30 - (
                int(time.time()) % 30
            )

            if remaining < 2:
                remaining = 2

            print(
                f"[angel_client] Waiting "
                f"{remaining} seconds for a fresh TOTP..."
            )

            time.sleep(remaining)

    # -----------------------------------------------------
    # All attempts failed
    # -----------------------------------------------------

    raise RuntimeError(
        f"Angel One login failed after "
        f"{max_attempts} attempts."
    )


# ---------------------------------------------------------
# Get SmartAPI client
# ---------------------------------------------------------

def get_client() -> SmartConnect:
    """
    Return the active SmartConnect client.

    If there is no active session, login automatically.
    """

    global _smart_api

    if _smart_api is None:
        login()

    return _smart_api


# ---------------------------------------------------------
# Feed token
# ---------------------------------------------------------

def get_feed_token() -> str:
    """
    Return Angel One Feed Token.

    This token is required for SmartStream WebSocket
    live market data.
    """

    client = get_client()

    return client.getfeedToken()
# ---------------------------------------------------------
# WebSocket auth token
# ---------------------------------------------------------
def get_auth_token() -> str:
    """Return the JWT auth token from the active SmartAPI session."""
    global _session_data
    if _session_data is None:
        login()
    try:
        return str(_session_data["data"]["jwtToken"])
    except Exception as e:
        raise RuntimeError(f"Angel auth token unavailable: {e}")
