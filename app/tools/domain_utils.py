"""
Domain extraction and validation utilities.
Handles URL parsing, domain cleaning, and validation.
"""

import re
import logging
from typing import Optional
from urllib.parse import urlparse
import tldextract

from ..config import BLACKLISTED_DOMAINS

logger = logging.getLogger(__name__)


def extract_clean_domain(url_or_text: str) -> Optional[str]:
    """
    Extract a clean domain from various input formats.
    
    Examples:
        "https://www.delhivery.com/about" → "delhivery.com"
        "www.zomato.com" → "zomato.com"
        "infosys.com" → "infosys.com"
        "Infosys Ltd" → None (not a domain)
    
    Args:
        url_or_text: URL, domain, or text that might contain a domain
        
    Returns:
        Clean domain or None if not found
    """
    if not url_or_text:
        return None
    
    text = url_or_text.strip()
    
    # Try to parse as URL first
    if "://" in text or text.startswith("www."):
        try:
            # Add protocol if missing
            if not text.startswith(("http://", "https://")):
                text = "https://" + text
            
            parsed = urlparse(text)
            hostname = parsed.netloc or parsed.path.split("/")[0]
            
            # Remove www prefix
            if hostname.startswith("www."):
                hostname = hostname[4:]
            
            # Validate it looks like a domain
            if "." in hostname and len(hostname) > 3:
                return hostname.lower()
        except Exception:
            pass
    
    # Try tldextract for more robust parsing
    try:
        extracted = tldextract.extract(text)
        if extracted.domain and extracted.suffix:
            domain = f"{extracted.domain}.{extracted.suffix}"
            return domain.lower()
    except Exception:
        pass
    
    # Try to find domain pattern in text
    domain_pattern = r'([a-zA-Z0-9][-a-zA-Z0-9]*\.)+[a-zA-Z]{2,}'
    match = re.search(domain_pattern, text)
    if match:
        domain = match.group(0).lower()
        # Remove www if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    
    return None


def is_valid_company_domain(domain: str) -> bool:
    """
    Check if a domain is a valid company domain.
    
    Filters out:
    - Blacklisted domains (social media, news sites, etc.)
    - Invalid TLDs
    - Too short domains
    
    Args:
        domain: Domain to validate
        
    Returns:
        True if domain is valid for email finding
    """
    if not domain:
        return False
    
    domain = domain.lower().strip()
    
    # Check minimum length
    if len(domain) < 4:
        return False
    
    # Check blacklist
    if domain in BLACKLISTED_DOMAINS:
        return False
    
    # Check for blacklisted base domains
    for blacklisted in BLACKLISTED_DOMAINS:
        if domain.endswith("." + blacklisted):
            return False
    
    # Validate has proper TLD
    try:
        extracted = tldextract.extract(domain)
        if not extracted.domain or not extracted.suffix:
            return False
        
        # Common valid TLDs for Indian companies
        valid_tlds = {
            "com", "in", "co.in", "io", "ai", "tech", "org", 
            "net", "co", "app", "dev", "cloud", "software"
        }
        
        # Allow any TLD for now, but log unusual ones
        if extracted.suffix not in valid_tlds:
            logger.debug(f"Unusual TLD for domain: {domain}")
        
        return True
    except Exception:
        return False


def extract_domain_from_company_name(company_name: str) -> Optional[str]:
    """
    Generate likely domain from company name.
    
    Examples:
        "Delhivery Pvt Ltd" → "delhivery.com"
        "Zomato" → "zomato.com"
        "Tata Consultancy Services" → "tcs.com"
    
    Args:
        company_name: Company name
        
    Returns:
        Likely domain or None
    """
    if not company_name:
        return None
    
    # Clean company name
    name = company_name.lower().strip()
    
    # Remove common suffixes
    suffixes_to_remove = [
        " private limited", " pvt ltd", " pvt. ltd.", " pvt ltd.",
        " limited", " ltd", " ltd.", " inc", " inc.",
        " llp", " llc", " corporation", " corp", " corp.",
        " india", " technologies", " solutions", " services",
        " software", " systems", " consulting"
    ]
    
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Remove special characters and extra spaces
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', '', name)  # Remove all spaces for domain
    
    if len(name) < 2:
        return None
    
    # Common Indian company domain patterns
    potential_domains = [
        f"{name}.com",
        f"{name}.in",
        f"{name}.co.in",
        f"{name}.io"
    ]
    
    return potential_domains[0]  # Return most likely (.com)


def normalize_domain(domain: str) -> str:
    """
    Normalize a domain to consistent format.
    
    Args:
        domain: Domain to normalize
        
    Returns:
        Normalized domain (lowercase, no www)
    """
    if not domain:
        return ""
    
    domain = domain.lower().strip()
    
    # Remove protocol
    domain = re.sub(r'^https?://', '', domain)
    
    # Remove www
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Remove trailing slash and path
    domain = domain.split("/")[0]
    
    return domain


def extract_domains_from_text(text: str) -> list:
    """
    Extract all domains from a text block.
    
    Args:
        text: Text that may contain domains
        
    Returns:
        List of unique valid domains
    """
    if not text:
        return []
    
    # Find all potential domains
    domain_pattern = r'(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*(?:\.[a-zA-Z0-9][-a-zA-Z0-9]*)+)'
    matches = re.findall(domain_pattern, text)
    
    # Clean and validate
    domains = []
    seen = set()
    
    for match in matches:
        domain = match.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        if domain not in seen and is_valid_company_domain(domain):
            seen.add(domain)
            domains.append(domain)
    
    return domains
