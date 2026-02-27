"""
Brevo (formerly Sendinblue) transactional email sender.

Safety features:
  - Global kill switch: EMAIL_SENDING_ENABLED must be True
  - Test mode: EMAIL_TEST_MODE=True redirects ALL emails to EMAIL_TEST_RECIPIENT
  - Dry-run capability: returns what WOULD be sent without actually sending
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

BREVO_API_URL = "https://api.brevo.com/v3/smtp/email"


class BrevoEmailResult:
    """Result of an email send attempt."""

    def __init__(
        self,
        success: bool,
        message_id: str = "",
        recipient: str = "",
        subject: str = "",
        error: str = "",
        test_mode: bool = False,
        dry_run: bool = False,
    ):
        self.success = success
        self.message_id = message_id
        self.recipient = recipient
        self.subject = subject
        self.error = error
        self.test_mode = test_mode
        self.dry_run = dry_run
        self.sent_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "message_id": self.message_id,
            "recipient": self.recipient,
            "subject": self.subject,
            "error": self.error,
            "test_mode": self.test_mode,
            "dry_run": self.dry_run,
            "sent_at": self.sent_at,
        }


class BrevoTool:
    """Send transactional emails via Brevo API."""

    def __init__(self):
        self.settings = get_settings()

    def _is_enabled(self) -> tuple[bool, str]:
        """Check if email sending is enabled. Returns (enabled, reason)."""
        if not self.settings.email_sending_enabled:
            return False, "EMAIL_SENDING_ENABLED is False"
        if not self.settings.brevo_api_key:
            return False, "BREVO_API_KEY not configured"
        if not self.settings.brevo_sender_email:
            return False, "BREVO_SENDER_EMAIL not configured"
        return True, ""

    def send_email(
        self,
        to_email: str,
        to_name: str,
        subject: str,
        body: str,
        dry_run: bool = False,
    ) -> BrevoEmailResult:
        """Send a transactional email via Brevo.

        Args:
            to_email: Recipient email address
            to_name: Recipient name
            subject: Email subject line
            body: Email body (plain text, converted to HTML paragraphs)
            dry_run: If True, validate everything but don't actually send

        Returns:
            BrevoEmailResult with success status, message_id, etc.
        """
        enabled, reason = self._is_enabled()
        if not enabled:
            return BrevoEmailResult(
                success=False,
                error=f"Email sending disabled: {reason}",
                recipient=to_email,
                subject=subject,
            )

        # Test mode: redirect to dummy recipient
        actual_recipient = to_email
        test_mode = self.settings.email_test_mode
        if test_mode:
            actual_recipient = self.settings.email_test_recipient
            logger.info(
                f"TEST MODE: Redirecting email from {to_email} → {actual_recipient}"
            )

        # Convert plain text body to simple HTML
        html_body = _text_to_html(body, to_name, subject, test_mode, to_email)

        payload = {
            "sender": {
                "name": self.settings.brevo_sender_name,
                "email": self.settings.brevo_sender_email,
            },
            "to": [{"email": actual_recipient, "name": to_name}],
            "subject": subject,
            "htmlContent": html_body,
            "textContent": body,
        }

        if dry_run:
            logger.info(f"DRY RUN: Would send to {actual_recipient}: {subject}")
            return BrevoEmailResult(
                success=True,
                recipient=actual_recipient,
                subject=subject,
                test_mode=test_mode,
                dry_run=True,
            )

        # Send via Brevo API
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    BREVO_API_URL,
                    headers={
                        "api-key": self.settings.brevo_api_key,
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                message_id = data.get("messageId", "")

            logger.info(
                f"Email sent via Brevo: {actual_recipient} | "
                f"subject='{subject[:40]}' | messageId={message_id} | "
                f"test_mode={test_mode}"
            )
            return BrevoEmailResult(
                success=True,
                message_id=message_id,
                recipient=actual_recipient,
                subject=subject,
                test_mode=test_mode,
            )

        except httpx.HTTPStatusError as e:
            error_body = ""
            try:
                error_body = e.response.text[:200]
            except Exception:
                pass
            error_msg = f"Brevo API error {e.response.status_code}: {error_body}"
            logger.error(error_msg)
            return BrevoEmailResult(
                success=False,
                error=error_msg,
                recipient=actual_recipient,
                subject=subject,
                test_mode=test_mode,
            )

        except Exception as e:
            error_msg = f"Email send failed: {str(e)[:200]}"
            logger.error(error_msg)
            return BrevoEmailResult(
                success=False,
                error=error_msg,
                recipient=actual_recipient,
                subject=subject,
                test_mode=test_mode,
            )


def _text_to_html(
    body: str,
    to_name: str,
    subject: str,
    test_mode: bool,
    original_email: str,
) -> str:
    """Convert plain text email body to simple HTML."""
    # Escape HTML
    import html as html_mod

    escaped = html_mod.escape(body)
    paragraphs = escaped.split("\n\n")
    html_paragraphs = "".join(
        f"<p style='margin:0 0 12px 0;line-height:1.5'>{p.replace(chr(10), '<br>')}</p>"
        for p in paragraphs
        if p.strip()
    )

    test_banner = ""
    if test_mode:
        test_banner = (
            f"<div style='background:#fff3cd;border:1px solid #ffc107;padding:10px;margin-bottom:16px;"
            f"border-radius:4px;font-size:12px;color:#856404'>"
            f"<strong>TEST MODE</strong> — This email was redirected. "
            f"Original recipient: {html_mod.escape(original_email)}"
            f"</div>"
        )

    return f"""<!DOCTYPE html>
<html><body style="font-family:Arial,sans-serif;font-size:14px;color:#333;max-width:600px;margin:0 auto;padding:20px">
{test_banner}
{html_paragraphs}
</body></html>"""
