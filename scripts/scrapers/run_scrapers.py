#!/usr/bin/env python3
"""CLI entry point for running documentation scrapers.

Usage:
    python scripts/scrapers/run_scrapers.py webhelp --version 25.5
    python scripts/scrapers/run_scrapers.py zendesk
    python scripts/scrapers/run_scrapers.py website --domain micromine.ru
    python scripts/scrapers/run_scrapers.py gkz
    python scripts/scrapers/run_scrapers.py geokniga
"""

import asyncio
import sys
from pathlib import Path

import click

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logging_config import setup_logging


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """Documentation scrapers for geology knowledge base."""
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    setup_logging(name="geo-scraper", level=level, log_dir=PROJECT_ROOT / "logs")


@cli.command()
@click.option("--version", default="25.5", help="Micromine version (e.g. 25.5)")
@click.option("--rate", default=2.0, help="Requests per second")
@click.option("--force", is_flag=True, help="Re-download all pages")
def webhelp(version: str, rate: float, force: bool) -> None:
    """Scrape webhelp.micromine.com documentation."""
    from scripts.scrapers.webhelp_scraper import WebhelpScraper

    scraper = WebhelpScraper(version=version, rate=rate)
    asyncio.run(scraper.run(force=force))


@cli.command()
@click.option("--rate", default=5.0, help="Requests per second")
@click.option("--force", is_flag=True, help="Re-download all articles")
def zendesk(rate: float, force: bool) -> None:
    """Scrape Micromine Zendesk knowledge base.

    Note: requires ZENDESK_API_TOKEN env var if the KB is access-restricted.
    """
    from scripts.scrapers.zendesk_scraper import ZendeskScraper

    scraper = ZendeskScraper(rate=rate)
    asyncio.run(scraper.run(force=force))


@cli.command()
@click.option(
    "--domain",
    default="micromine.ru",
    type=click.Choice(["micromine.ru", "micromine.kz"]),
    help="Domain to crawl",
)
@click.option("--rate", default=1.0, help="Requests per second")
@click.option("--force", is_flag=True, help="Re-download all pages")
def website(domain: str, rate: float, force: bool) -> None:
    """Crawl micromine.ru or micromine.kz product pages."""
    from scripts.scrapers.website_scraper import WebsiteScraper

    scraper = WebsiteScraper(domain=domain, rate=rate)
    asyncio.run(scraper.run(force=force))


@cli.command()
@click.option("--rate", default=1.0, help="Requests per second (be polite to gov site)")
@click.option("--force", is_flag=True, help="Re-download all documents")
def gkz(rate: float, force: bool) -> None:
    """Scrape ГКЗ regulatory documents from gkz-rf.ru."""
    from scripts.scrapers.gkz_scraper import GkzScraper

    scraper = GkzScraper(rate=rate)
    asyncio.run(scraper.run(force=force))


@cli.command()
@click.option("--rate", default=0.5, help="Requests per second (be polite)")
@click.option("--force", is_flag=True, help="Re-download all books")
def geokniga(rate: float, force: bool) -> None:
    """Download geological books from geokniga.org (Закревский et al.)."""
    from scripts.scrapers.geokniga_scraper import GeoknigaScraper

    scraper = GeoknigaScraper(rate=rate)
    asyncio.run(scraper.run(force=force))


if __name__ == "__main__":
    cli()
