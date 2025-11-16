"""
Site Metrics Calculator - Centralized calculation logic

This module provides a single source of truth for calculating
coverage, density, profit, and demand categories for charging stations.
Used by both visualization and CSV generation to ensure consistency.

Author: EVCS Optimization Team
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def calculate_site_metrics(site_idx: int, 
                         selected_sites: np.ndarray,
                         candidate_sites: pd.DataFrame,
                         demand_zones: pd.DataFrame,
                         distance_matrix: np.ndarray,
                         prices: np.ndarray) -> Dict:
    """
    Calculate all metrics for a single charging station site.
    
    This function ensures consistent calculations across visualization
    and CSV generation.
    
    Parameters:
    -----------
    site_idx : int
        Index of the site in candidate_sites
    selected_sites : np.ndarray
        Binary array indicating selected sites
    candidate_sites : pd.DataFrame
        DataFrame of all candidate sites
    demand_zones : pd.DataFrame
        DataFrame of all demand zones
    distance_matrix : np.ndarray
        Distance matrix (n_zones, n_sites)
    prices : np.ndarray
        Price array for all sites
        
    Returns:
    --------
    Dict
        Dictionary with all calculated metrics
    """
    site = candidate_sites.iloc[site_idx]
    price = float(prices[site_idx]) if site_idx < len(prices) else 10.0
    
    n_zones = len(demand_zones)
    
    # Calculate coverage (demand within 5km)
    coverage = 0.0
    density_sum = 0.0
    min_distance_to_demand = np.inf
    zones_in_range = 0
    
    # First, find average demand density from zones with actual demand
    zones_with_demand = []
    for i in range(n_zones):
        zone_demand = float(demand_zones.iloc[i]['demand'])
        if zone_demand == 0 and 'population' in demand_zones.columns:
            population = float(demand_zones.iloc[i]['population'])
            if population > 0:
                ev_density = float(demand_zones.iloc[i].get('ev_density', 0.03))
                zone_demand = population * ev_density
        if zone_demand > 0:
            zones_with_demand.append(zone_demand)
    
    avg_demand_density = np.mean(zones_with_demand) if zones_with_demand else 1.0
    baseline_demand_per_zone = avg_demand_density * 0.5
    
    # Iterate through all demand zones
    for i in range(n_zones):
        if i >= distance_matrix.shape[0] or site_idx >= distance_matrix.shape[1]:
            continue
            
        dist = float(distance_matrix[i, site_idx])
        
        if np.isnan(dist) or dist < 0:
            continue
            
        min_distance_to_demand = min(min_distance_to_demand, dist)
        
        if dist <= 5.0:  # 5km service radius
            zone_demand = float(demand_zones.iloc[i]['demand'])
            
            # If demand is zero, try to estimate from population
            if zone_demand == 0 and 'population' in demand_zones.columns:
                population = float(demand_zones.iloc[i]['population'])
                if population > 0:
                    ev_density = float(demand_zones.iloc[i].get('ev_density', 0.03))
                    zone_demand = population * ev_density
            
            # If still zero, use spatial interpolation
            if zone_demand == 0:
                nearby_demand = []
                for k in range(n_zones):
                    if k == i:
                        continue
                    other_dist = float(distance_matrix[k, site_idx])
                    if other_dist <= 10.0:
                        other_demand = float(demand_zones.iloc[k]['demand'])
                        if other_demand == 0 and 'population' in demand_zones.columns:
                            other_pop = float(demand_zones.iloc[k]['population'])
                            if other_pop > 0:
                                other_ev_density = float(demand_zones.iloc[k].get('ev_density', 0.03))
                                other_demand = other_pop * other_ev_density
                        
                        if other_demand > 0:
                            weight = 1.0 / (1.0 + other_dist)
                            nearby_demand.append(other_demand * weight)
                
                if nearby_demand:
                    zone_demand = np.mean(nearby_demand)
                else:
                    zone_demand = baseline_demand_per_zone
            
            if zone_demand > 0:
                coverage += zone_demand
                density_sum += zone_demand
                zones_in_range += 1
    
    # Calculate density
    if coverage > 0 and zones_in_range > 0:
        coverage_area_km2 = np.pi * 5.0 * 5.0
        density = density_sum / coverage_area_km2
    else:
        density = 0.0
    
    # Handle partial coverage for sites just outside range
    if coverage == 0 and min_distance_to_demand < np.inf:
        if min_distance_to_demand <= 7.0:
            nearest_demand = 0.0
            for i in range(n_zones):
                if i >= distance_matrix.shape[0] or site_idx >= distance_matrix.shape[1]:
                    continue
                dist = float(distance_matrix[i, site_idx])
                if np.isnan(dist) or dist < 0:
                    continue
                if dist <= 7.0:
                    zone_demand = float(demand_zones.iloc[i]['demand'])
                    if zone_demand == 0 and 'population' in demand_zones.columns:
                        population = float(demand_zones.iloc[i]['population'])
                        if population > 0:
                            ev_density = float(demand_zones.iloc[i].get('ev_density', 0.03))
                            zone_demand = population * ev_density
                    
                    if zone_demand > 0:
                        weight = max(0, 1 - (dist - 5.0) / 2.0)
                        nearest_demand += zone_demand * weight
            coverage = nearest_demand
            if coverage > 0:
                density = coverage / (np.pi * 7.0 * 7.0)
    
    # Calculate profit
    utilization_rate = 0.3
    if coverage > 0:
        utilization_rate = min(0.5, 0.2 + (coverage / 100.0) * 0.3)
    
    estimated_demand = coverage * utilization_rate
    
    # Revenue calculation
    charging_sessions_per_ev_per_month = 12.0
    kwh_per_session = 12.5
    monthly_kwh_per_ev = charging_sessions_per_ev_per_month * kwh_per_session
    
    actual_served_evs = estimated_demand
    monthly_revenue = actual_served_evs * monthly_kwh_per_ev * price
    
    # Operating cost
    electricity_cost_per_kwh = 4.0
    monthly_operating_cost = actual_served_evs * monthly_kwh_per_ev * electricity_cost_per_kwh
    monthly_maintenance = 5000.0
    
    # Monthly and annual profit
    monthly_profit = monthly_revenue - monthly_operating_cost - monthly_maintenance
    annual_profit = monthly_profit * 12
    
    # Categorize demand
    if coverage >= 100:
        demand_category = 'High'
    elif coverage >= 50:
        demand_category = 'Medium'
    elif coverage > 0:
        demand_category = 'Low'
    else:
        demand_category = 'Very Low'
    
    # Get location name
    location_name = site.get('name', '')
    if not location_name and 'site_type' in site:
        location_name = str(site['site_type'])
    
    return {
        'site_id': int(site['site_id']),
        'location_name': location_name,
        'latitude': float(site['latitude']),
        'longitude': float(site['longitude']),
        'coverage': coverage,
        'density': density,
        'annual_profit': annual_profit,
        'price': price,
        'capacity': int(site['capacity']),
        'setup_cost': float(site['setup_cost']),
        'total_setup_cost': float(site.get('total_setup_cost', site['setup_cost'])),
        'grid_upgrade_cost': float(site.get('grid_upgrade_cost', 0.0)),
        'grid_available_kw': float(site.get('grid_available_kw', 0.0)),
        'grid_required_kw': float(site.get('grid_required_kw', 0.0)),
        'grid_capacity_gap_kw': float(site.get('grid_capacity_gap_kw', 0.0)),
        'distance_to_grid_km': float(site.get('distance_to_grid_km', np.nan)),
        'grid_voltage_kv': float(site.get('grid_voltage_kv', np.nan)) if not pd.isnull(site.get('grid_voltage_kv', np.nan)) else np.nan,
        'nearest_grid_id': int(site.get('nearest_grid_id', -1)),
        'nearest_grid_name': site.get('nearest_grid_name', ''),
        'grid_capacity_ok': bool(site.get('grid_capacity_ok', True)),
        'site_type': site.get('site_type', 'unknown'),
        'demand_category': demand_category,
        'min_distance': min_distance_to_demand
    }

