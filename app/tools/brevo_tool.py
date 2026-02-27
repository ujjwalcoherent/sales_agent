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
        tone: str = "consultative",
        trend_title: str = "",
        company_name: str = "",
        lead_type: str = "",
    ) -> BrevoEmailResult:
        """Send a transactional email via Brevo.

        Args:
            to_email: Recipient email address
            to_name: Recipient name
            subject: Email subject line
            body: Email body (plain text, converted to branded HTML)
            dry_run: If True, validate everything but don't actually send
            tone: Outreach tone — "executive" | "consultative" | "professional"
            trend_title: Market trend that triggered this outreach
            company_name: Recipient's company name
            lead_type: "pain" | "opportunity" | "risk" | "intelligence"

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

        # Build branded HTML email
        html_body = build_branded_email(
            body=body,
            to_name=to_name,
            subject=subject,
            tone=tone,
            trend_title=trend_title,
            company_name=company_name,
            lead_type=lead_type,
            sender_name=self.settings.brevo_sender_name,
            sender_email=self.settings.brevo_sender_email,
            test_mode=test_mode,
            original_email=to_email,
        )

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


# ── Branded Email Templates ──────────────────────────────────────────
#
# Three templates matching the frontend design language:
#   executive    — Minimal, warm-toned accent bar. For C-suite.
#   consultative — Full branded header with trend context. For influencers.
#   professional — Clean corporate layout. For gatekeepers / formal intros.
#
# Color palette (from frontend globals.css light mode):
#   Accent/Amber: #B07030    Green: #2D6A4F    Blue: #2A5A8A
#   Red: #A83226             Text: #18170F     Surface: #FFFFFF
#   Background: #F8F7F2      Border: #E4E2D8

TONE_STYLES = {
    "executive": {
        "accent": "#B07030",
        "accent_light": "#F5EDE4",
        "accent_mid": "#D4A06A",
    },
    "consultative": {
        "accent": "#2A5A8A",
        "accent_light": "#E8EFF6",
        "accent_mid": "#6A9AC8",
    },
    "professional": {
        "accent": "#2D6A4F",
        "accent_light": "#E6F0EB",
        "accent_mid": "#5A9A78",
    },
}

LEAD_TYPE_COLORS = {
    "pain": "#A83226",
    "opportunity": "#2D6A4F",
    "risk": "#B07030",
    "intelligence": "#2A5A8A",
}


def build_branded_email(
    body: str,
    to_name: str,
    subject: str,
    tone: str = "consultative",
    trend_title: str = "",
    company_name: str = "",
    lead_type: str = "",
    sender_name: str = "Coherent Market Insights",
    sender_email: str = "",
    test_mode: bool = False,
    original_email: str = "",
) -> str:
    """Build a branded HTML email matching the CMI frontend design system.

    Card-style layout: rounded container, no top bar, logo-integrated
    accent color, trend context pill, branded footer with gradient.
    """
    import html as html_mod

    style = TONE_STYLES.get(tone, TONE_STYLES["consultative"])
    accent = style["accent"]

    # Escape and format body
    escaped = html_mod.escape(body)
    paragraphs = escaped.split("\n\n")
    html_paragraphs = "".join(
        f"<p style='margin:0 0 16px 0;line-height:1.75;color:#2C2B23;font-size:14px'>"
        f"{p.replace(chr(10), '<br>')}</p>"
        for p in paragraphs
        if p.strip()
    )

    # Test mode banner (outside the card, above it)
    test_banner = ""
    if test_mode:
        test_banner = (
            f"<div style='max-width:560px;margin:0 auto 12px;padding:8px 16px;"
            f"background:#FFF8E1;border:1px solid #FFD54F;border-radius:8px;"
            f"font-size:11px;color:#8D6E00;text-align:center'>"
            f"<strong>TEST MODE</strong> &mdash; Redirected from "
            f"<span style=\"background:#FFF3CD;padding:1px 6px;border-radius:3px;font-size:10px;"
            f"font-family:monospace\">{html_mod.escape(original_email)}</span></div>"
        )

    # Recipient context line under the logo
    context_line = ""
    if company_name:
        context_line = (
            f"<div style='font-size:11px;color:#8A8878;margin-top:10px;letter-spacing:0.01em'>"
            f"{html_mod.escape(to_name)} &middot; {html_mod.escape(company_name)}</div>"
        )

    # Logo URL (287x60 horizontal wordmark — scales to 150x31 in header, 110x23 in footer)
    logo_url = "https://www.coherentmarketinsights.com/images/indexv2/cmi-admin-logo.png"

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#F0EFE8;font-family:'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif">

<!-- Spacer top -->
<div style="height:28px"></div>

{test_banner}

<!-- Card container -->
<div style="max-width:560px;margin:0 auto;background:#FFFFFF;border-radius:12px;overflow:hidden;
     box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04)">

  <!-- Header: logo + context -->
  <div style="padding:28px 36px 22px">
    <a href="https://www.coherentmarketinsights.com" style="text-decoration:none">
      <img src="{logo_url}"
           alt="{html_mod.escape(sender_name)}" width="150" height="31"
           style="width:150px;height:auto;display:block"
      />
    </a>
    {context_line}
  </div>

  <!-- Divider -->
  <div style="margin:0 36px;height:1px;background:#E8E6DC"></div>

  <!-- Body -->
  <div style="padding:24px 36px 28px">
    {html_paragraphs}
  </div>

  <!-- Footer -->
  <div style="padding:18px 36px;background:#FAFAF6;border-top:1px solid #EDEBE2">
    <table cellpadding="0" cellspacing="0" border="0" style="width:100%"><tr>
      <td style="vertical-align:middle">
        <img src="{logo_url}"
             alt="CMI" width="110" height="23"
             style="width:110px;height:auto;display:block;opacity:0.55"
        />
      </td>
      <td style="vertical-align:middle;text-align:right">
        <a href="https://www.coherentmarketinsights.com"
           style="display:inline-block;font-size:11px;font-weight:600;color:{accent};
                  text-decoration:none;padding:6px 14px;border:1px solid {accent}33;
                  border-radius:6px">
          Visit Website &rarr;
        </a>
      </td>
    </tr></table>
  </div>

</div>

<!-- Compliance footer -->
<div style="max-width:560px;margin:28px auto 28px;padding:0 36px;background:#F0EFE8;text-align:center;font-size:10px;color:#BBB;line-height:1.8">
  {html_mod.escape(sender_name)} &bull; Market Intelligence &amp; Advisory<br>
  <a href="mailto:{html_mod.escape(sender_email)}?subject=Unsubscribe" style="color:#BBB;text-decoration:underline">Unsubscribe</a>
</div>

</body></html>"""
