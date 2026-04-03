"""
Fertigation Recommendation System
Rule-based engine for fertilizer and irrigation scheduling.
Based on soil moisture, crop type, and growth stage.
"""

# Soil moisture thresholds (%)
MOISTURE_THRESHOLDS = {
    "low": 20,
    "medium": 40,
    "high": 60,
}

# Crop water requirements (mm/day) by growth stage
CROP_WATER_NEEDS = {
    "rice": {"seedling": 5.0, "vegetative": 8.0, "flowering": 10.0, "fruiting": 7.0, "maturity": 4.0},
    "wheat": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.0, "maturity": 2.5},
    "maize": {"seedling": 3.5, "vegetative": 5.5, "flowering": 7.0, "fruiting": 6.0, "maturity": 3.0},
    "cotton": {"seedling": 2.5, "vegetative": 5.0, "flowering": 7.5, "fruiting": 6.5, "maturity": 3.5},
    "sugarcane": {"seedling": 4.0, "vegetative": 7.0, "flowering": 9.0, "fruiting": 8.0, "maturity": 5.0},
    "potato": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.0, "fruiting": 5.5, "maturity": 3.0},
    "tomato": {"seedling": 2.5, "vegetative": 4.5, "flowering": 6.0, "fruiting": 5.5, "maturity": 3.5},
    "soybean": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.0, "maturity": 2.5},
    "chickpea": {"seedling": 2.0, "vegetative": 3.5, "flowering": 5.0, "fruiting": 3.5, "maturity": 1.5},
    "lentil": {"seedling": 2.0, "vegetative": 3.5, "flowering": 5.0, "fruiting": 3.5, "maturity": 1.5},
    "banana": {"seedling": 4.0, "vegetative": 7.0, "flowering": 9.0, "fruiting": 8.0, "maturity": 5.0},
    "mango": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.5, "maturity": 3.0},
    "grapes": {"seedling": 2.5, "vegetative": 4.5, "flowering": 6.0, "fruiting": 5.0, "maturity": 3.0},
    "apple": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.5, "maturity": 3.0},
    "orange": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.5, "maturity": 3.0},
    "coconut": {"seedling": 3.5, "vegetative": 5.5, "flowering": 7.0, "fruiting": 6.0, "maturity": 4.0},
    "papaya": {"seedling": 3.0, "vegetative": 5.0, "flowering": 6.5, "fruiting": 5.5, "maturity": 3.5},
    "coffee": {"seedling": 2.5, "vegetative": 4.5, "flowering": 6.0, "fruiting": 5.0, "maturity": 3.0},
    "pomegranate": {"seedling": 2.5, "vegetative": 4.5, "flowering": 6.0, "fruiting": 5.0, "maturity": 3.0},
    "kidneybeans": {"seedling": 2.0, "vegetative": 4.0, "flowering": 5.5, "fruiting": 4.0, "maturity": 2.0},
    "pigeonpeas": {"seedling": 2.0, "vegetative": 3.5, "flowering": 5.0, "fruiting": 4.0, "maturity": 2.0},
    "mothbeans": {"seedling": 1.5, "vegetative": 3.0, "flowering": 4.0, "fruiting": 3.0, "maturity": 1.5},
    "mungbean": {"seedling": 2.0, "vegetative": 3.5, "flowering": 5.0, "fruiting": 3.5, "maturity": 1.5},
    "watermelon": {"seedling": 3.0, "vegetative": 5.0, "flowering": 7.0, "fruiting": 6.0, "maturity": 3.5},
    "muskmelon": {"seedling": 3.0, "vegetative": 5.0, "flowering": 7.0, "fruiting": 6.0, "maturity": 3.5},
}

# Fertilizer recommendations (kg/hectare) by crop and growth stage
FERTILIZER_SCHEDULE = {
    "rice": {
        "seedling": {"N": 30, "P": 20, "K": 20, "name": "Starter NPK (10-26-26)"},
        "vegetative": {"N": 60, "P": 0, "K": 30, "name": "Urea + MOP"},
        "flowering": {"N": 40, "P": 0, "K": 20, "name": "Top-dress Urea"},
        "fruiting": {"N": 20, "P": 0, "K": 10, "name": "Maintenance dose"},
        "maturity": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
    },
    "wheat": {
        "seedling": {"N": 40, "P": 30, "K": 20, "name": "Basal DAP + Urea"},
        "vegetative": {"N": 50, "P": 0, "K": 0, "name": "First top-dress Urea"},
        "flowering": {"N": 30, "P": 0, "K": 15, "name": "Second top-dress"},
        "fruiting": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
        "maturity": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
    },
    "maize": {
        "seedling": {"N": 35, "P": 25, "K": 25, "name": "Basal NPK (15-15-15)"},
        "vegetative": {"N": 60, "P": 0, "K": 30, "name": "Side-dress Urea + MOP"},
        "flowering": {"N": 35, "P": 0, "K": 15, "name": "Top-dress at tasseling"},
        "fruiting": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
        "maturity": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
    },
    "default": {
        "seedling": {"N": 30, "P": 20, "K": 15, "name": "Starter NPK (10-26-26)"},
        "vegetative": {"N": 50, "P": 10, "K": 25, "name": "Growth fertilizer (20-10-10)"},
        "flowering": {"N": 30, "P": 15, "K": 20, "name": "Bloom booster (10-30-20)"},
        "fruiting": {"N": 20, "P": 10, "K": 30, "name": "Fruit enhancer (10-10-30)"},
        "maturity": {"N": 0, "P": 0, "K": 0, "name": "No fertilizer needed"},
    },
}


def get_moisture_status(moisture_percent: float) -> str:
    """Classify soil moisture level."""
    if moisture_percent < MOISTURE_THRESHOLDS["low"]:
        return "CRITICAL - Severely Dry"
    elif moisture_percent < MOISTURE_THRESHOLDS["medium"]:
        return "LOW - Needs Irrigation"
    elif moisture_percent < MOISTURE_THRESHOLDS["high"]:
        return "ADEQUATE - Moderate Moisture"
    else:
        return "HIGH - Well Hydrated"


def get_irrigation_recommendation(crop: str, moisture_percent: float, growth_stage: str) -> dict:
    """Generate irrigation recommendation based on inputs."""
    crop_lower = crop.lower().strip()
    
    water_needs = CROP_WATER_NEEDS.get(crop_lower, CROP_WATER_NEEDS.get("wheat", {}))
    water_required = water_needs.get(growth_stage, 5.0)
    
    moisture_status = get_moisture_status(moisture_percent)
    
    if moisture_percent < MOISTURE_THRESHOLDS["low"]:
        irrigation_amount = water_required * 1.5
        urgency = "IMMEDIATE"
        frequency = "Daily"
    elif moisture_percent < MOISTURE_THRESHOLDS["medium"]:
        irrigation_amount = water_required
        urgency = "SOON (within 24h)"
        frequency = "Every 2-3 days"
    elif moisture_percent < MOISTURE_THRESHOLDS["high"]:
        irrigation_amount = water_required * 0.5
        urgency = "MONITOR"
        frequency = "Every 4-5 days"
    else:
        irrigation_amount = 0
        urgency = "NONE"
        frequency = "No irrigation needed"
    
    return {
        "crop": crop.title(),
        "growth_stage": growth_stage.title(),
        "moisture_status": moisture_status,
        "water_required_mm": round(water_required, 2),
        "irrigation_amount_mm": round(irrigation_amount, 2),
        "urgency": urgency,
        "frequency": frequency,
    }


def get_fertilizer_recommendation(crop: str, growth_stage: str) -> dict:
    """Generate fertilizer recommendation based on crop and growth stage."""
    crop_lower = crop.lower().strip()
    
    schedule = FERTILIZER_SCHEDULE.get(crop_lower, FERTILIZER_SCHEDULE.get("default", {}))
    if not schedule:
        return {
            "crop": crop.title(),
            "growth_stage": growth_stage.title(),
            "fertilizer_name": "General NPK",
            "nitrogen_kg_ha": 0,
            "phosphorus_kg_ha": 0,
            "potassium_kg_ha": 0,
            "total_npk_kg_ha": 0,
        }
    
    fert_info = schedule.get(growth_stage, schedule.get("vegetative", {}))
    if not fert_info:
        fert_info = {"name": "General NPK", "N": 0, "P": 0, "K": 0}
    
    return {
        "crop": crop.title(),
        "growth_stage": growth_stage.title(),
        "fertilizer_name": fert_info.get("name", "General NPK"),
        "nitrogen_kg_ha": fert_info.get("N", 0),
        "phosphorus_kg_ha": fert_info.get("P", 0),
        "potassium_kg_ha": fert_info.get("K", 0),
        "total_npk_kg_ha": fert_info.get("N", 0) + fert_info.get("P", 0) + fert_info.get("K", 0),
    }


def get_combined_recommendation(crop: str, moisture_percent: float, growth_stage: str) -> dict:
    """Get combined fertigation (fertilizer + irrigation) recommendation."""
    irrigation = get_irrigation_recommendation(crop, moisture_percent, growth_stage)
    fertilizer = get_fertilizer_recommendation(crop, growth_stage)
    
    return {
        "irrigation": irrigation,
        "fertilizer": fertilizer,
    }
