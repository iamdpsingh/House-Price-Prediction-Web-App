"""
Utility Functions for House Price Prediction API
Input validation, data processing, and response formatting
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Union
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate input data for house price prediction
    
    Args:
        data: Dictionary containing input parameters
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Required fields
    required_fields = ['location', 'total_sqft', 'bhk', 'bath']
    for field in required_fields:
        if field not in data or data[field] is None or str(data[field]).strip() == '':
            errors.append(f'{field.replace("_", " ").title()} is required')
    
    if errors:  # Return early if required fields are missing
        return False, errors
    
    # Location validation
    location = str(data['location']).strip()
    if len(location) < 2:
        errors.append('Location must be at least 2 characters long')
    
    # Total square feet validation
    try:
        total_sqft = float(data['total_sqft'])
        if total_sqft < 300:
            errors.append('Total square feet must be at least 300')
        elif total_sqft > 10000:
            errors.append('Total square feet cannot exceed 10,000')
    except (ValueError, TypeError):
        errors.append('Total square feet must be a valid number')
    
    # BHK validation
    try:
        bhk = int(data['bhk'])
        if bhk < 1:
            errors.append('BHK must be at least 1')
        elif bhk > 10:
            errors.append('BHK cannot exceed 10')
    except (ValueError, TypeError):
        errors.append('BHK must be a valid integer')
    
    # Bathroom validation
    try:
        bath = int(data['bath'])
        if bath < 1:
            errors.append('Number of bathrooms must be at least 1')
        elif bath > 10:
            errors.append('Number of bathrooms cannot exceed 10')
    except (ValueError, TypeError):
        errors.append('Number of bathrooms must be a valid integer')
    
    # Balcony validation (optional field)
    if 'balcony' in data and data['balcony'] is not None:
        try:
            balcony = int(data['balcony'])
            if balcony < 0:
                errors.append('Number of balconies cannot be negative')
            elif balcony > 10:
                errors.append('Number of balconies cannot exceed 10')
        except (ValueError, TypeError):
            errors.append('Number of balconies must be a valid integer')
    
    # Cross-field validation
    if not errors:  # Only if individual fields are valid
        try:
            bhk = int(data['bhk'])
            bath = int(data['bath'])
            total_sqft = float(data['total_sqft'])
            
            # Bathroom to BHK ratio check
            if bath > bhk + 3:
                errors.append(f'Too many bathrooms ({bath}) for {bhk} BHK property')
            
            # Square feet per room check
            sqft_per_room = total_sqft / bhk
            if sqft_per_room < 200:
                errors.append('Square feet per room seems too small (minimum 200 sqft per room recommended)')
            elif sqft_per_room > 2000:
                errors.append('Square feet per room seems unusually large')
                
        except (ValueError, TypeError):
            pass  # Individual field errors already caught above
    
    return len(errors) == 0, errors

def clean_location_name(location: str) -> str:
    """
    Clean and standardize location names
    
    Args:
        location: Raw location string
        
    Returns:
        Cleaned location string
    """
    if not isinstance(location, str):
        return str(location)
    
    # Basic cleaning
    location = location.strip()
    location = re.sub(r'\s+', ' ', location)  # Replace multiple spaces with single space
    
    # Remove special characters but keep spaces and hyphens
    location = re.sub(r'[^\w\s\-]', '', location)
    
    # Title case
    location = location.title()
    
    # Common standardizations
    standardizations = {
        'Electronic City': 'Electronic City',
        'E City': 'Electronic City',
        'Ecity': 'Electronic City',
        'Whitefield': 'Whitefield',
        'White Field': 'Whitefield',
        'Sarjapur Road': 'Sarjapur Road',
        'Sarjapura Road': 'Sarjapur Road',
        'Bannerghatta Road': 'Bannerghatta Road',
        'Bannerghatta Rd': 'Bannerghatta Road',
        'Marathahalli': 'Marathahalli',
        'Marathalli': 'Marathahalli',
        'Kr Puram': 'KR Puram',
        'K R Puram': 'KR Puram'
    }
    
    for variant, standard in standardizations.items():
        if location.lower() == variant.lower():
            return standard
    
    return location

def calculate_derived_metrics(predicted_price: float, total_sqft: float, bhk: int, 
                            bath: int, balcony: int) -> Dict[str, Any]:
    """
    Calculate additional metrics from prediction results
    
    Args:
        predicted_price: Predicted price in lakhs
        total_sqft: Total square feet
        bhk: Number of BHK
        bath: Number of bathrooms  
        balcony: Number of balconies
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    try:
        # Price per square feet (in rupees)
        if total_sqft > 0:
            metrics['price_per_sqft'] = (predicted_price * 100000) / total_sqft
        else:
            metrics['price_per_sqft'] = 0
        
        # Price per room
        if bhk > 0:
            metrics['price_per_room'] = (predicted_price * 100000) / bhk
        else:
            metrics['price_per_room'] = 0
        
        # Room efficiency metrics
        if bhk > 0:
            metrics['sqft_per_room'] = total_sqft / bhk
            metrics['bath_per_room'] = bath / bhk
        else:
            metrics['sqft_per_room'] = 0
            metrics['bath_per_room'] = 0
        
        # Amenity score (simple scoring system)
        amenity_score = 0
        amenity_score += min(bath * 10, 30)  # Max 30 points for bathrooms
        amenity_score += min(balcony * 5, 15)  # Max 15 points for balconies
        amenity_score += min((bhk - 1) * 5, 20)  # Max 20 points for additional rooms
        metrics['amenity_score'] = min(amenity_score, 65)  # Cap at 65
        
    except Exception as e:
        logger.error(f"Error calculating derived metrics: {e}")
        # Return default values on error
        metrics = {
            'price_per_sqft': 0,
            'price_per_room': 0,
            'sqft_per_room': 0,
            'bath_per_room': 0,
            'amenity_score': 0
        }
    
    return metrics

def get_price_category(price: float) -> Dict[str, str]:
    """
    Categorize price into market segments
    
    Args:
        price: Price in lakhs
        
    Returns:
        Dictionary with category and description
    """
    if price < 30:
        return {
            'category': 'Budget',
            'description': 'Affordable housing segment',
            'color': '#28a745'  # Green
        }
    elif price < 60:
        return {
            'category': 'Mid-range',
            'description': 'Middle-class housing segment',
            'color': '#17a2b8'  # Blue
        }
    elif price < 100:
        return {
            'category': 'Premium',
            'description': 'Upper-middle-class segment',
            'color': '#ffc107'  # Yellow
        }
    elif price < 200:
        return {
            'category': 'Luxury',
            'description': 'High-end luxury segment',
            'color': '#fd7e14'  # Orange
        }
    else:
        return {
            'category': 'Ultra-luxury',
            'description': 'Ultra-premium segment',
            'color': '#e83e8c'  # Pink
        }

def format_currency(amount: float, currency: str = '₹', unit: str = 'lakhs') -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Numeric amount
        currency: Currency symbol
        unit: Unit description
        
    Returns:
        Formatted currency string
    """
    try:
        if amount >= 100:  # Convert to crores if >= 100 lakhs
            crores = amount / 100
            return f"{currency}{crores:.2f} crores"
        else:
            return f"{currency}{amount:.2f} {unit}"
    except:
        return f"{currency}0.00 {unit}"

def format_number(number: Union[int, float], decimals: int = 0) -> str:
    """
    Format number with Indian numbering system
    
    Args:
        number: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    try:
        if decimals == 0:
            return f"{int(number):,}"
        else:
            return f"{number:,.{decimals}f}"
    except:
        return str(number)

def get_market_insights(location: str, predicted_price: float, 
                       property_details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate market insights based on prediction
    
    Args:
        location: Property location
        predicted_price: Predicted price
        property_details: Property characteristics
        
    Returns:
        Dictionary of market insights
    """
    insights = {
        'location_tier': 'Unknown',
        'investment_potential': 'Moderate',
        'market_position': 'Average',
        'recommendations': []
    }
    
    try:
        # Location tier analysis (simplified)
        premium_locations = [
            'Koramangala', 'Indiranagar', 'HSR Layout', 'Jayanagar',
            'Malleshwaram', 'Whitefield', 'Electronic City'
        ]
        
        emerging_locations = [
            'Sarjapur Road', 'Marathahalli', 'Bannerghatta Road',
            'Yelahanka', 'Thanisandra', 'KR Puram'
        ]
        
        if location in premium_locations:
            insights['location_tier'] = 'Premium'
            insights['investment_potential'] = 'Stable'
        elif location in emerging_locations:
            insights['location_tier'] = 'Emerging'
            insights['investment_potential'] = 'High Growth'
        else:
            insights['location_tier'] = 'Developing'
            insights['investment_potential'] = 'Moderate Growth'
        
        # Market position based on price
        if predicted_price < 50:
            insights['market_position'] = 'Below Average'
            insights['recommendations'].append('Good entry point for first-time buyers')
        elif predicted_price < 100:
            insights['market_position'] = 'Average'
            insights['recommendations'].append('Balanced investment option')
        else:
            insights['market_position'] = 'Above Average'
            insights['recommendations'].append('Premium property with good amenities')
        
        # Property-specific recommendations
        bhk = property_details.get('bhk', 0)
        total_sqft = property_details.get('total_sqft', 0)
        
        if bhk >= 3 and total_sqft >= 1400:
            insights['recommendations'].append('Suitable for families')
        
        if bhk <= 2 and total_sqft <= 1200:
            insights['recommendations'].append('Ideal for young professionals')
        
    except Exception as e:
        logger.error(f"Error generating market insights: {e}")
    
    return insights

def format_response(prediction_result: Dict[str, Any], 
                   input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the complete API response
    
    Args:
        prediction_result: Result from model prediction
        input_data: Original input data
        
    Returns:
        Formatted response dictionary
    """
    try:
        predicted_price = prediction_result['predicted_price']
        
        # Calculate additional metrics
        derived_metrics = calculate_derived_metrics(
            predicted_price,
            input_data['total_sqft'],
            input_data['bhk'],
            input_data['bath'],
            input_data.get('balcony', 0)
        )
        
        # Get price category
        price_category_info = get_price_category(predicted_price)
        
        # Get market insights
        market_insights = get_market_insights(
            input_data['location'],
            predicted_price,
            input_data
        )
        
        # Format the response
        response = {
            'success': True,
            'prediction': {
                'predicted_price': predicted_price,
                'formatted_price': format_currency(predicted_price),
                'price_range': {
                    'min': round(predicted_price * 0.9, 2),
                    'max': round(predicted_price * 1.1, 2)
                }
            },
            'property_details': {
                'location': clean_location_name(input_data['location']),
                'total_sqft': input_data['total_sqft'],
                'bhk': input_data['bhk'],
                'bath': input_data['bath'],
                'balcony': input_data.get('balcony', 0)
            },
            'metrics': {
                'price_per_sqft': round(derived_metrics['price_per_sqft'], 2),
                'price_per_sqft_formatted': format_currency(
                    derived_metrics['price_per_sqft'] / 100000, '₹', 'per sqft'
                ),
                'sqft_per_room': round(derived_metrics['sqft_per_room'], 1),
                'amenity_score': derived_metrics['amenity_score']
            },
            'category': {
                'name': price_category_info['category'],
                'description': price_category_info['description'],
                'color': price_category_info['color']
            },
            'market_insights': market_insights,
            'model_info': {
                'model_used': prediction_result.get('model_used', 'Unknown'),
                'confidence': prediction_result.get('confidence', 'Medium')
            },
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error formatting response: {e}")
        return {
            'success': False,
            'error': 'Failed to format prediction response',
            'timestamp': pd.Timestamp.now().isoformat()
        }

def log_prediction_request(input_data: Dict[str, Any], 
                          prediction_result: Dict[str, Any] = None,
                          error: str = None) -> None:
    """
    Log prediction requests for monitoring and analytics
    
    Args:
        input_data: Input parameters
        prediction_result: Prediction results (if successful)
        error: Error message (if failed)
    """
    try:
        log_entry = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'location': input_data.get('location'),
            'total_sqft': input_data.get('total_sqft'),
            'bhk': input_data.get('bhk'),
            'bath': input_data.get('bath'),
            'balcony': input_data.get('balcony', 0)
        }
        
        if prediction_result:
            log_entry.update({
                'predicted_price': prediction_result.get('predicted_price'),
                'model_used': prediction_result.get('model_used'),
                'success': True
            })
            logger.info(f"Prediction successful: {log_entry}")
        else:
            log_entry.update({
                'error': error,
                'success': False
            })
            logger.warning(f"Prediction failed: {log_entry}")
            
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

# Validation helper functions
def is_valid_sqft(sqft: Union[int, float]) -> bool:
    """Check if square feet value is valid"""
    try:
        sqft = float(sqft)
        return 300 <= sqft <= 10000
    except:
        return False

def is_valid_bhk(bhk: Union[int, str]) -> bool:
    """Check if BHK value is valid"""
    try:
        bhk = int(bhk)
        return 1 <= bhk <= 10
    except:
        return False

def is_valid_bath(bath: Union[int, str]) -> bool:
    """Check if bathroom count is valid"""
    try:
        bath = int(bath)
        return 1 <= bath <= 10
    except:
        return False

def is_valid_balcony(balcony: Union[int, str]) -> bool:
    """Check if balcony count is valid"""
    try:
        balcony = int(balcony)
        return 0 <= balcony <= 10
    except:
        return False

# Testing functions
if __name__ == "__main__":
    # Test validation function
    test_data = {
        'location': 'Whitefield',
        'total_sqft': 1200,
        'bhk': 2,
        'bath': 2,
        'balcony': 1
    }
    
    is_valid, errors = validate_input_data(test_data)
    print(f"Validation test: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        print(f"Errors: {errors}")
    
    # Test response formatting
    mock_prediction = {
        'predicted_price': 85.5,
        'model_used': 'random_forest',
        'confidence': 'High'
    }
    
    response = format_response(mock_prediction, test_data)
    print(f"Response formatting test: {'PASSED' if response['success'] else 'FAILED'}")
