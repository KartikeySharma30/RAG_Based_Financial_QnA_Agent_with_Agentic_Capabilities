"""
Document Processor
Handles text extraction, table conversion to markdown, and image captioning
"""

import os
import re
import base64
import requests
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup, NavigableString, Tag
import pandas as pd
from PIL import Image
import io
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from urllib.parse import urljoin, urlparse
import html2text

class DocumentProcessor:
    """Processes SEC filing documents to extract and format content"""
    
    def __init__(self):
        # Initialize image captioning model
        self.caption_processor = None
        self.caption_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # HTML to markdown converter
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = False
        self.html_converter.body_width = 0
        
    def _load_caption_model(self):
        """Load image captioning model lazily"""
        if self.caption_processor is None:
            print("Loading image captioning model...")
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model.to(self.device)
    
    def extract_images(self, soup: BeautifulSoup, base_url: str = "") -> List[Dict]:
        """Extract images from HTML and generate captions"""
        images = []
        img_tags = soup.find_all('img')
        
        if not img_tags:
            return images
        
        self._load_caption_model()
        
        for i, img_tag in enumerate(img_tags):
            try:
                src = img_tag.get('src', '')
                alt = img_tag.get('alt', '')
                
                if not src:
                    continue
                
                # Handle relative URLs
                if src.startswith('/'):
                    if base_url:
                        img_url = urljoin(base_url, src)
                    else:
                        continue
                elif not src.startswith(('http://', 'https://')):
                    if base_url:
                        img_url = urljoin(base_url, src)
                    else:
                        continue
                else:
                    img_url = src
                
                # Download and process image
                caption = self._generate_image_caption(img_url, alt)
                
                if caption:
                    images.append({
                        'index': i,
                        'src': src,
                        'url': img_url,
                        'alt': alt,
                        'caption': caption,
                        'element': img_tag
                    })
                    
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
        
        return images
    
    def _generate_image_caption(self, img_url: str, alt_text: str = "") -> Optional[str]:
        """Generate caption for an image"""
        try:
            # Download image
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            
            # Open image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check if image looks like a chart/graph (basic heuristic)
            width, height = image.size
            if width < 50 or height < 50:  # Too small to be meaningful
                return None
            
            # Generate caption
            inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
            out = self.caption_model.generate(**inputs, max_length=100, num_beams=3)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            # Enhanced caption for financial charts/graphs
            if any(keyword in alt_text.lower() for keyword in ['chart', 'graph', 'revenue', 'income', 'profit', 'growth']):
                caption = f"Financial chart/graph: {caption}"
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption: {str(e)}")
            return alt_text if alt_text else None
    
    def extract_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract tables and convert to markdown"""
        tables = []
        table_tags = soup.find_all('table')
        
        for i, table_tag in enumerate(table_tags):
            try:
                # Convert table to pandas DataFrame
                df = self._table_to_dataframe(table_tag)
                
                if df is not None and not df.empty:
                    # Convert to markdown
                    markdown = df.to_markdown(index=False, tablefmt='pipe')
                    
                    tables.append({
                        'index': i,
                        'dataframe': df,
                        'markdown': markdown,
                        'element': table_tag
                    })
                    
            except Exception as e:
                print(f"Error processing table {i}: {str(e)}")
                continue
        
        return tables
    
    def _table_to_dataframe(self, table_tag: Tag) -> Optional[pd.DataFrame]:
        """Convert HTML table to pandas DataFrame"""
        try:
            # Extract rows
            rows = []
            
            # Get header row
            header_row = table_tag.find('tr')
            if header_row:
                headers = []
                for th in header_row.find_all(['th', 'td']):
                    headers.append(th.get_text(strip=True))
                
                if headers:
                    rows.append(headers)
            
            # Get data rows
            for tr in table_tag.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all(['td', 'th']):
                    cell_text = td.get_text(strip=True)
                    # Clean up financial numbers
                    cell_text = re.sub(r'\s+', ' ', cell_text)
                    row.append(cell_text)
                
                if row:
                    rows.append(row)
            
            if len(rows) < 2:  # Need at least header and one data row
                return None
            
            # Create DataFrame
            df = pd.DataFrame(rows[1:], columns=rows[0])
            
            # Clean up DataFrame
            df = df.dropna(how='all')  # Remove empty rows
            df = df.loc[:, ~df.columns.duplicated()]  # Remove duplicate columns
            
            return df
            
        except Exception as e:
            print(f"Error converting table to DataFrame: {str(e)}")
            return None
    
    def extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def process_document(self, filepath: str, base_url: str = "") -> Dict:
        """Process a complete document and merge all content"""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        
        print(f"Processing document: {os.path.basename(filepath)}")
        
        # Extract components
        print("  - Extracting tables...")
        tables = self.extract_tables(soup)
        
        print("  - Extracting images...")
        images = self.extract_images(soup, base_url)
        
        print("  - Processing text content...")
        processed_content = self._merge_content(soup, tables, images)
        
        return {
            'filepath': filepath,
            'original_html': content,
            'processed_content': processed_content,
            'tables': tables,
            'images': images,
            'soup': soup
        }
    
    def _merge_content(self, soup: BeautifulSoup, tables: List[Dict], images: List[Dict]) -> str:
        """Merge text, tables, and image captions while preserving structure"""
        # Create copies of the soup to avoid modifying original
        working_soup = BeautifulSoup(str(soup), 'html.parser')
        
        # Replace tables with markdown
        for table_info in tables:
            table_element = working_soup.find('table')  # Find first table (they get processed in order)
            if table_element:
                # Create markdown representation
                markdown_text = f"\n\n{table_info['markdown']}\n\n"
                table_element.replace_with(markdown_text)
        
        # Replace images with captions
        for img_info in images:
            img_elements = working_soup.find_all('img')
            for img_element in img_elements:
                if img_element.get('src') == img_info['src']:
                    caption_text = f"\n\n[IMAGE: {img_info['caption']}]\n\n"
                    img_element.replace_with(caption_text)
                    break
        
        # Convert to clean text
        text = self.extract_text_content(working_soup)
        
        return text

# Example usage and testing
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Test with a sample HTML file
    test_file = "/workspace/data/sec_filings/GOOGL_10K_2023.html"
    if os.path.exists(test_file):
        result = processor.process_document(test_file)
        print(f"Processed {len(result['tables'])} tables and {len(result['images'])} images")
        print(f"Content length: {len(result['processed_content'])} characters")