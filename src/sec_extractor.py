"""
SEC Edgar Data Extractor
Fetches 10-K annual filings for specified companies and years
"""

import os
import requests
import json
import time
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta

class SECExtractor:
    """Extracts SEC filings from Edgar database"""
    
    def __init__(self):
        self.base_url = "https://www.sec.gov/Archives/edgar/data"
        self.search_url = "https://data.sec.gov/submissions"
        self.headers = {
            'User-Agent': 'Financial RAG Assistant v1.0 (contact@example.com)',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        
        # Company ticker to CIK mapping
        self.company_ciks = {
            'GOOGL': '0001652044',  # Alphabet Inc.
            'MSFT': '0000789019',   # Microsoft Corporation
            'NVDA': '0001045810'    # NVIDIA Corporation
        }
    
    def get_company_submissions(self, ticker: str) -> Dict:
        """Get all submissions for a company"""
        cik = self.company_ciks.get(ticker)
        if not cik:
            raise ValueError(f"Unknown ticker: {ticker}")
        
        url = f"{self.search_url}/CIK{cik}.json"
        
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        # SEC rate limiting
        time.sleep(0.1)
        
        return response.json()
    
    def find_10k_filings(self, ticker: str, years: List[int]) -> List[Dict]:
        """Find 10-K filings for specified years"""
        submissions = self.get_company_submissions(ticker)
        
        filings = []
        recent_filings = submissions.get('filings', {}).get('recent', {})
        
        if not recent_filings:
            return filings
        
        forms = recent_filings.get('form', [])
        filing_dates = recent_filings.get('filingDate', [])
        accession_numbers = recent_filings.get('accessionNumber', [])
        primary_documents = recent_filings.get('primaryDocument', [])
        
        for i, form in enumerate(forms):
            if form == '10-K':
                filing_date = filing_dates[i]
                filing_year = int(filing_date.split('-')[0])
                
                if filing_year in years:
                    filings.append({
                        'ticker': ticker,
                        'form': form,
                        'filing_date': filing_date,
                        'year': filing_year,
                        'accession_number': accession_numbers[i],
                        'primary_document': primary_documents[i]
                    })
        
        return filings
    
    def download_filing(self, filing_info: Dict, save_dir: str) -> str:
        """Download a specific 10-K filing"""
        ticker = filing_info['ticker']
        year = filing_info['year']
        accession_number = filing_info['accession_number'].replace('-', '')
        primary_document = filing_info['primary_document']
        cik = self.company_ciks[ticker]
        
        # Construct URL
        url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession_number}/{primary_document}"
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Download file
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        
        # Save file
        filename = f"{ticker}_10K_{year}.html"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded: {filename}")
        
        # SEC rate limiting
        time.sleep(0.1)
        
        return filepath
    
    def extract_all_filings(self, companies: List[str], years: List[int], save_dir: str) -> List[str]:
        """Extract 10-K filings for all specified companies and years"""
        downloaded_files = []
        
        for company in companies:
            print(f"Processing {company}...")
            
            try:
                filings = self.find_10k_filings(company, years)
                
                for filing in filings:
                    filepath = self.download_filing(filing, save_dir)
                    downloaded_files.append(filepath)
                    
            except Exception as e:
                print(f"Error processing {company}: {str(e)}")
                continue
        
        return downloaded_files

# Example usage and testing
if __name__ == "__main__":
    extractor = SECExtractor()
    
    # Test extraction
    companies = ['GOOGL', 'MSFT', 'NVDA']
    years = [2022, 2023, 2024]
    save_dir = "/workspace/data/sec_filings"
    
    files = extractor.extract_all_filings(companies, years, save_dir)
    print(f"Downloaded {len(files)} files")