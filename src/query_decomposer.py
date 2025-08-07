"""
Query Decomposer
Decomposes complex queries into simpler sub-queries for better retrieval
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    """Types of financial queries"""
    SIMPLE_DIRECT = "simple_direct"
    COMPARATIVE = "comparative" 
    CROSS_COMPANY = "cross_company"
    TEMPORAL = "temporal"
    CALCULATION = "calculation"

@dataclass
class SubQuery:
    """Represents a decomposed sub-query"""
    text: str
    query_type: QueryType
    company: Optional[str] = None
    year: Optional[int] = None
    metric: Optional[str] = None
    section_hint: Optional[str] = None

@dataclass
class DecomposedQuery:
    """Result of query decomposition"""
    original_query: str
    query_type: QueryType
    sub_queries: List[SubQuery]
    requires_calculation: bool = False
    calculation_type: Optional[str] = None

class QueryDecomposer:
    """Decomposes financial queries into manageable sub-queries"""
    
    def __init__(self):
        # Company mappings and aliases
        self.company_mappings = {
            'google': 'GOOGL',
            'alphabet': 'GOOGL', 
            'microsoft': 'MSFT',
            'nvidia': 'NVDA',
            'googl': 'GOOGL',
            'msft': 'MSFT',
            'nvda': 'NVDA'
        }
        
        # Financial metrics patterns
        self.financial_metrics = {
            'revenue': ['revenue', 'sales', 'total revenue', 'net sales'],
            'profit': ['profit', 'net income', 'earnings', 'net profit'],
            'operating_income': ['operating income', 'operating profit', 'operating earnings'],
            'operating_margin': ['operating margin', 'operating profit margin'],
            'gross_margin': ['gross margin', 'gross profit margin'],
            'cash': ['cash', 'cash and equivalents', 'cash position'],
            'debt': ['debt', 'total debt', 'long-term debt'],
            'assets': ['assets', 'total assets'],
            'equity': ['equity', 'shareholders equity', 'stockholders equity'],
            'eps': ['earnings per share', 'eps'],
            'data_center_revenue': ['data center revenue', 'datacenter revenue', 'data center sales']
        }
        
        # Temporal keywords
        self.temporal_keywords = {
            'growth': ['growth', 'grew', 'increase', 'increased', 'change'],
            'decline': ['decline', 'decreased', 'drop', 'fell'],
            'comparison': ['compare', 'comparison', 'versus', 'vs', 'compared to']
        }
        
        # Comparison keywords
        self.comparison_keywords = [
            'highest', 'lowest', 'best', 'worst', 'most', 'least',
            'greater', 'better', 'which company', 'who had'
        ]
    
    def extract_companies(self, query: str) -> List[str]:
        """Extract company tickers from query"""
        companies = []
        query_lower = query.lower()
        
        for name, ticker in self.company_mappings.items():
            if name in query_lower:
                if ticker not in companies:
                    companies.append(ticker)
        
        return companies
    
    def extract_years(self, query: str) -> List[int]:
        """Extract years from query"""
        years = []
        year_matches = re.findall(r'\b(20\d{2})\b', query)
        
        for year_str in year_matches:
            year = int(year_str)
            if 2020 <= year <= 2025:  # Reasonable range
                years.append(year)
        
        return list(set(years))  # Remove duplicates
    
    def extract_metrics(self, query: str) -> List[str]:
        """Extract financial metrics from query"""
        metrics = []
        query_lower = query.lower()
        
        for metric_key, metric_variants in self.financial_metrics.items():
            for variant in metric_variants:
                if variant in query_lower:
                    if metric_key not in metrics:
                        metrics.append(metric_key)
                    break
        
        return metrics
    
    def classify_query_type(self, query: str, companies: List[str], years: List[int]) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        # Check for cross-company analysis
        if len(companies) > 1 or any(keyword in query_lower for keyword in self.comparison_keywords):
            return QueryType.CROSS_COMPANY
        
        # Check for comparative/temporal analysis
        if len(years) > 1 or any(keyword in query_lower for temporal_list in self.temporal_keywords.values() for keyword in temporal_list):
            return QueryType.COMPARATIVE
        
        # Check for calculation requirements
        if any(keyword in query_lower for keyword in ['calculate', 'compute', 'growth rate', 'percentage']):
            return QueryType.CALCULATION
        
        # Default to simple direct query
        return QueryType.SIMPLE_DIRECT
    
    def decompose_simple_query(self, query: str, companies: List[str], years: List[int], metrics: List[str]) -> List[SubQuery]:
        """Decompose simple direct queries"""
        # For simple queries, usually no decomposition needed
        company = companies[0] if companies else None
        year = years[0] if years else None
        metric = metrics[0] if metrics else None
        
        return [SubQuery(
            text=query,
            query_type=QueryType.SIMPLE_DIRECT,
            company=company,
            year=year,
            metric=metric
        )]
    
    def decompose_comparative_query(self, query: str, companies: List[str], years: List[int], metrics: List[str]) -> List[SubQuery]:
        """Decompose comparative queries (temporal analysis)"""
        sub_queries = []
        
        if len(companies) == 1 and len(years) > 1 and len(metrics) >= 1:
            company = companies[0]
            metric = metrics[0]
            
            # Create sub-queries for each year
            for year in years:
                sub_query_text = f"{company} {metric} {year}"
                sub_queries.append(SubQuery(
                    text=sub_query_text,
                    query_type=QueryType.SIMPLE_DIRECT,
                    company=company,
                    year=year,
                    metric=metric
                ))
            
            # Add calculation instruction
            if len(years) == 2:
                calculation_text = f"Calculate growth from {min(years)} to {max(years)}"
                sub_queries.append(SubQuery(
                    text=calculation_text,
                    query_type=QueryType.CALCULATION,
                    company=company,
                    metric=metric
                ))
        
        return sub_queries
    
    def decompose_cross_company_query(self, query: str, companies: List[str], years: List[int], metrics: List[str]) -> List[SubQuery]:
        """Decompose cross-company analysis queries"""
        sub_queries = []
        
        year = years[0] if years else None
        metric = metrics[0] if metrics else None
        
        # If specific companies mentioned, use them
        if companies:
            target_companies = companies
        else:
            # Default to all three companies
            target_companies = ['GOOGL', 'MSFT', 'NVDA']
        
        # Create sub-queries for each company
        for company in target_companies:
            sub_query_text = f"{company} {metric} {year}" if metric and year else f"{company} {query}"
            sub_queries.append(SubQuery(
                text=sub_query_text,
                query_type=QueryType.SIMPLE_DIRECT,
                company=company,
                year=year,
                metric=metric
            ))
        
        # Add comparison instruction
        comparison_text = f"Compare {metric} across companies" if metric else "Compare across companies"
        sub_queries.append(SubQuery(
            text=comparison_text,
            query_type=QueryType.CALCULATION,
            metric=metric
        ))
        
        return sub_queries
    
    def decompose_query(self, query: str) -> DecomposedQuery:
        """Main method to decompose a query"""
        # Extract entities
        companies = self.extract_companies(query)
        years = self.extract_years(query)
        metrics = self.extract_metrics(query)
        
        # Classify query type
        query_type = self.classify_query_type(query, companies, years)
        
        # Decompose based on type
        if query_type == QueryType.SIMPLE_DIRECT:
            sub_queries = self.decompose_simple_query(query, companies, years, metrics)
            requires_calculation = False
            
        elif query_type == QueryType.COMPARATIVE:
            sub_queries = self.decompose_comparative_query(query, companies, years, metrics)
            requires_calculation = True
            
        elif query_type == QueryType.CROSS_COMPANY:
            sub_queries = self.decompose_cross_company_query(query, companies, years, metrics)
            requires_calculation = True
            
        else:  # CALCULATION type
            sub_queries = self.decompose_simple_query(query, companies, years, metrics)
            requires_calculation = True
        
        # Determine calculation type
        calculation_type = None
        if requires_calculation:
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ['growth', 'grew', 'increase']):
                calculation_type = 'growth'
            elif any(keyword in query_lower for keyword in self.comparison_keywords):
                calculation_type = 'comparison'
            else:
                calculation_type = 'general'
        
        return DecomposedQuery(
            original_query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            requires_calculation=requires_calculation,
            calculation_type=calculation_type
        )
    
    def get_query_examples(self) -> Dict[str, List[str]]:
        """Get example queries for each type"""
        return {
            "Simple Direct": [
                "What was Microsoft's total revenue in 2023?",
                "NVIDIA's data center revenue in 2024",
                "Google's operating margin 2022"
            ],
            "Comparative": [
                "How did NVIDIA's data center revenue grow from 2022 to 2023?",
                "Microsoft's revenue change between 2022 and 2024",
                "Google's profit growth from 2023 to 2024"
            ],
            "Cross-Company": [
                "Which company had the highest operating margin in 2023?",
                "Compare revenue across all three companies in 2024",
                "Who had the best profit margins in 2022?"
            ]
        }

# Example usage and testing
if __name__ == "__main__":
    decomposer = QueryDecomposer()
    
    # Test queries
    test_queries = [
        "What was Microsoft's total revenue in 2023?",
        "How did NVIDIA's data center revenue grow from 2022 to 2023?", 
        "Which company had the highest operating margin in 2023?",
        "Compare Google and Microsoft revenue in 2024"
    ]
    
    for query in test_queries:
        print(f"\nOriginal Query: {query}")
        result = decomposer.decompose_query(query)
        print(f"Type: {result.query_type.value}")
        print(f"Requires Calculation: {result.requires_calculation}")
        print("Sub-queries:")
        for i, sub_query in enumerate(result.sub_queries):
            print(f"  {i+1}. {sub_query.text} (Company: {sub_query.company}, Year: {sub_query.year})")
        print("-" * 50)