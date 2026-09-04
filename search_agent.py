#!/usr/bin/env python3
import sys
import re
import html
import datetime
from email.utils import parsedate_to_datetime
import requests
import xml.etree.ElementTree as ET

# ANSI Color codes for beautiful terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Feed configurations
FEEDS = [
    {
        "name": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "type": "rss"
    },
    {
        "name": "The Verge",
        "url": "https://www.theverge.com/rss/index.xml",
        "type": "atom"
    },
    {
        "name": "Wired",
        "url": "https://www.wired.com/feed/rss",
        "type": "rss"
    },
    {
        "name": "Hacker News",
        "url": "https://news.ycombinator.com/rss",
        "type": "rss"
    }
]

def clean_html(raw_html):
    """Strips HTML tags and decodes HTML entities to get clean text."""
    if not raw_html:
        return ""
    # Strip HTML tags
    clean_text = re.sub(r'<[^>]+>', ' ', raw_html)
    # Decode HTML entities (e.g., &amp;, &quot;, &#8217;)
    clean_text = html.unescape(clean_text)
    # Clean up whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def get_sentence_summary(text, max_sentences=3):
    """Extracts the first few sentences of a text to create a concise summary."""
    if not text:
        return "No summary available."
    # Simple sentence splitting regex (handles ., !, ? followed by space or end of line)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty or extremely short sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return text[:150] + "..." if len(text) > 150 else text
    summary = " ".join(sentences[:max_sentences])
    return summary

def parse_date(date_str, feed_type):
    """Parses date string into a timezone-aware datetime object."""
    if not date_str:
        return None
    try:
        if feed_type == "rss":
            # RSS typically uses RFC 822 format (e.g., "Tue, 15 Aug 2026 12:34:56 +0000")
            return parsedate_to_datetime(date_str)
        elif feed_type == "atom":
            # Atom typically uses ISO 8601 format (e.g., "2026-08-15T12:34:56-04:00")
            # Replace Z with +00:00 for fromisoformat compatibility in some Python versions
            if date_str.endswith('Z'):
                date_str = date_str[:-1] + '+00:00'
            return datetime.datetime.fromisoformat(date_str)
    except Exception as e:
        # Fallback if parsing fails
        return None
    return None

def fetch_feed(feed_info):
    """Fetches and parses a single RSS or Atom feed."""
    name = feed_info["name"]
    url = feed_info["url"]
    feed_type = feed_info["type"]
    
    print(f"{Colors.BLUE}Fetching {name}...{Colors.ENDC}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
    except Exception as e:
        print(f"{Colors.FAIL}Error fetching {name}: {e}{Colors.ENDC}\n")
        return []

    articles = []
    try:
        root = ET.fromstring(response.content)
        
        if feed_type == "rss":
            # RSS 2.0 structure: channel -> item
            items = root.findall('.//item')
            for item in items:
                title = item.findtext('title')
                link = item.findtext('link')
                pub_date_str = item.findtext('pubDate')
                
                # Try getting description, fallback to content:encoded if available
                description = item.findtext('description')
                if not description:
                    # Some feeds use content namespace
                    content_encoded = item.find('{http://purl.org/rss/1.0/modules/content/}encoded')
                    if content_encoded is not None:
                        description = content_encoded.text
                
                pub_date = parse_date(pub_date_str, "rss")
                
                articles.append({
                    "source": name,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "description": description
                })
                
        elif feed_type == "atom":
            # Atom structure uses namespaces
            # Retrieve default namespace
            ns = {'ns': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('ns:entry', ns)
            if not entries:
                # If namespace is different or not declared, try a generic search
                entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                
            for entry in entries:
                title_elem = entry.find('ns:title', ns)
                if title_elem is None:
                    title_elem = entry.find('.//{http://www.w3.org/2005/Atom}title')
                title = title_elem.text if title_elem is not None else ""
                
                link_elem = entry.find('ns:link', ns)
                if link_elem is None:
                    link_elem = entry.find('.//{http://www.w3.org/2005/Atom}link')
                link = link_elem.get('href') if link_elem is not None else ""
                
                published_elem = entry.find('ns:published', ns)
                if published_elem is None:
                    published_elem = entry.find('ns:updated', ns)
                if published_elem is None:
                    published_elem = entry.find('.//{http://www.w3.org/2005/Atom}published')
                pub_date_str = published_elem.text if published_elem is not None else ""
                
                summary_elem = entry.find('ns:summary', ns)
                if summary_elem is None:
                    summary_elem = entry.find('ns:content', ns)
                if summary_elem is None:
                    summary_elem = entry.find('.//{http://www.w3.org/2005/Atom}summary')
                description = summary_elem.text if summary_elem is not None else ""
                
                pub_date = parse_date(pub_date_str, "atom")
                
                articles.append({
                    "source": name,
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "description": description
                })
                
    except Exception as e:
        print(f"{Colors.FAIL}Error parsing XML for {name}: {e}{Colors.ENDC}\n")
        
    return articles

def format_relative_time(dt):
    """Formats datetime to relative time string (e.g., '3 hours ago')."""
    if not dt:
        return "Unknown date"
    
    now = datetime.datetime.now(dt.tzinfo)
    diff = now - dt
    
    seconds = diff.total_seconds()
    if seconds < 0:
        return "Just now"
    
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    days = int(diff.days)
    
    if days > 0:
        return f"{days} day{'s' if days > 1 else ''} ago"
    elif hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif minutes > 0:
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"

def main():
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== ULUSLARARASI TEKNOLOJİ HABERLERİ AJANI ==={Colors.ENDC}")
    print(f"{Colors.CYAN}Son 24 saat içindeki en güncel haberler derleniyor...\n{Colors.ENDC}")
    
    # Define the 24 hour threshold (using timezone-aware comparison)
    now = datetime.datetime.now(datetime.timezone.utc)
    one_day_ago = now - datetime.timedelta(hours=24)
    
    all_articles = []
    
    for feed_info in FEEDS:
        articles = fetch_feed(feed_info)
        filtered_count = 0
        
        for art in articles:
            pub_date = art["pub_date"]
            # Filter for last 24 hours
            if pub_date:
                # Convert pub_date to UTC for comparison
                pub_date_utc = pub_date.astimezone(datetime.timezone.utc)
                if pub_date_utc >= one_day_ago:
                    all_articles.append(art)
                    filtered_count += 1
            else:
                # If no date, exclude to satisfy 24-hour constraint strictly
                pass
                
        print(f"{Colors.GREEN}Found {filtered_count} articles from the last 24 hours.{Colors.ENDC}\n")
    
    # Sort all articles by publication date (newest first)
    all_articles.sort(key=lambda x: x["pub_date"] if x["pub_date"] else datetime.datetime.min.replace(tzinfo=datetime.timezone.utc), reverse=True)
    
    print(f"{Colors.HEADER}{Colors.BOLD}=== GÜNCEL TEKNOLOJİ ÖZETLERİ ({len(all_articles)} Haber) ==={Colors.ENDC}\n")
    
    if not all_articles:
        print(f"{Colors.WARNING}Son 24 saat içinde yayınlanmış haber bulunamadı.{Colors.ENDC}\n")
        return

    for idx, art in enumerate(all_articles, 1):
        clean_title = clean_html(art["title"])
        raw_desc = art["description"]
        clean_desc = clean_html(raw_desc)
        summary = get_sentence_summary(clean_desc, max_sentences=3)
        relative_time = format_relative_time(art["pub_date"])
        
        # Source tag colors
        source_color = Colors.CYAN
        if art["source"] == "TechCrunch":
            source_color = Colors.GREEN
        elif art["source"] == "The Verge":
            source_color = Colors.WARNING
        elif art["source"] == "Wired":
            source_color = Colors.HEADER
        
        print(f"{Colors.BOLD}{idx}. {clean_title}{Colors.ENDC}")
        print(f"   [{source_color}{art['source']}{Colors.ENDC}] | {Colors.BLUE}{relative_time}{Colors.ENDC}")
        print(f"   {Colors.BOLD}Özet:{Colors.ENDC} {summary}")
        print(f"   {Colors.UNDERLINE}Link:{Colors.ENDC} {art['link']}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}İşlem kullanıcı tarafından iptal edildi.{Colors.ENDC}")
        sys.exit(0)
