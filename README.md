# Grain Silo Ventilation Control System

A Streamlit web application that helps determine safe ventilation conditions for grain storage based on temperature, humidity, and dew point calculations.

## System Overview

The system consists of two main components:

### Part A: Streamlit UI & Data Handler
- **ExcelParser**: Reads and validates Excel files with temperature/humidity data
- **DataFormatter**: Converts data between UI and logic engine formats
- **StreamlitUI**: User interface for file upload, configuration, and results display

### Part B: Ventilation Logic Engine
- **PhysicsCalculator**: Dew point, grain temperature, and intergranular RH calculations
- **ConditionChecker**: Implements ventilation safety conditions C1-C4
- **VentilationController**: Main logic engine with `evaluate()` method

## Installation

1. **Clone or download the project files**
2. **Install dependencies:**
   ```bash
   cd ventilation_system
   pip install -r requirements.txt
   ```

## Running the Application

```bash
cd ventilation_system
streamlit run part_a.py
```

The application will open in your web browser at `http://localhost:8501`

## Excel Data Format

Upload Excel files (.xlsx) with the following structure:

### Required Columns
- **Temperature**: Any of these column names:
  - `temp`, `temperature`, `t`, `temp_c`, `temperature_c`, `air_temp`
- **Humidity**: Any of these column names:
  - `rh`, `humidity`, `relative_humidity`, `humidity_percent`, `rel_hum`

### Data Requirements
- At least 24 hourly readings (first 24 values will be used)
- Temperature range: -50°C to 60°C
- Humidity range: 0% to 100%

### Example Excel Structure
| Hour | Temperature(°C) | Relative Humidity(%) |
|------|----------------|---------------------|
| 0    | 15.2           | 78.5                |
| 1    | 14.8           | 79.1                |
| ...  | ...            | ...                 |
| 23   | 16.1           | 76.3                |

## Ventilation Conditions

The system evaluates four safety conditions:

### C1: Dew Point Check
- **Requirement**: Dew point must be below grain temperature
- **Purpose**: Prevents condensation on grain during ventilation

### C2: Humidity Check  
- **Requirement**: Intergranular RH must be below 75%
- **Purpose**: Prevents mold growth and spoilage

### C3: Temperature Difference
- **Requirement**: Temperature difference must exceed 3°C
- **Purpose**: Ensures sufficient thermal driving force for ventilation

### C4: Layer Temperature Difference
- **Requirement**: Layer temperature difference must exceed 5°C
- **Purpose**: Reduces temperature stratification

## Ventilation Logic

The fan control decision follows this logic:

```
If ventilation_mode = "unventilated":
    Fan = OFF
Else if ventilation_mode in ["regular", "automated"]:
    If (C1 AND C2) OR (C3 OR C4):
        Fan = ON
    Else:
        Fan = OFF
```

## Configuration Options

### Silo Selection
- **Silo One**: No insulation
- **Silo Two**: 4cm foam insulation (temperature factor: 0.8)

### Ventilation Modes
- **Unventilated**: Fan always OFF
- **Regular**: Automatic fan control based on conditions
- **Automated**: Same logic as regular mode

### Current Hour
- Select hour (0-23) for evaluation
- Defaults to current system hour

## Sample Test Files

Three sample Excel files are provided:

1. **`sample_data_safe.xlsx`**: Cool, dry conditions → Fan ON
2. **`sample_data_unsafe.xlsx`**: Warm, humid conditions → Fan OFF  
3. **`sample_data_mixed.xlsx`**: Varying conditions → Depends on hour

## Testing

### Unit Tests
```bash
# Test Part B logic engine
python -m unittest test_part_b.py -v
```

### Integration Tests
```bash
# Run comprehensive tests
python -c "
from part_b import VentilationController
controller = VentilationController()
# ... test scenarios
"
```

## System Architecture

```
┌─────────────────────────────────────────┐
│             STREAMLIT WEB APP          │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────────┐         ┌───────────┐ │
│  │   PART A:    │────────>│ PART B:    │ │
│  │  UI & Data   │  JSON   │ Logic      │ │
│  │  Handler     │ payload │ Engine     │ │
│  └──────────────┘<────────└───────────┘ │
│        │                              │ │
│   Excel Upload                    Fan    │ │
│   User Inputs                     Decision│ │
│   Display Results                  Output │ │
│                                         │
└─────────────────────────────────────────┘
```

## Technical Implementation

### Part B Constants
```python
INTERGRANULAR_RH_THRESHOLD = 75.0  # percent
TEMP_DIFF_C3_THRESHOLD = 3.0       # °C
TEMP_DIFF_C4_THRESHOLD = 5.0       # °C
INSULATION_FACTOR = 0.8            # for silo two
```

### Dew Point Calculation
Uses Magnus-Tetens approximation (Alduchov & Eskridge, 1996):
- α = 17.625, β = 243.04°C
- Accuracy: ±0.35°C for -40°C to +50°C range

### JSON Interface

#### Input to Part B
```json
{
  "silo": "one" | "two",
  "ventilation_mode": "unventilated" | "regular" | "automated", 
  "external_temp": [float],  // 24 hourly values
  "external_rh": [float],    // 24 hourly values
  "current_hour": int        // 0-23
}
```

#### Output from Part B
```json
{
  "fan_status": "ON" | "OFF",
  "reason": str,
  "conditions_met": ["C1", "C2"], 
  "timestamp": str
}
```

## Error Handling

- **Invalid Excel files**: Clear error messages with file preview
- **Missing columns**: Shows expected column names
- **Invalid data**: Specifies validation requirements
- **System errors**: Graceful degradation with user feedback

## Limitations

1. **Simplified Models**: Uses simplified psychrometric relationships
2. **Layer Temperature**: C4 uses estimation based on daily range
3. **Environmental Factors**: Doesn't account for wind, solar radiation, etc.
4. **Assumptions**: assumes uniform grain properties

## Future Enhancements

1. **Real sensor integration**: Connect to actual silo monitoring systems
2. **Advanced models**: More complex moisture transfer calculations
3. **Historical data**: Store and analyze ventilation decisions over time
4. **Multi-silo management**: Handle multiple silos simultaneously
5. **Alerts**: Notification system for critical conditions

## Support

For technical assistance or questions:
- Check system logs for error details
- Verify Excel file format meets requirements
- Ensure all dependencies are properly installed

---

**Disclaimer**: This system provides recommendations based on sensor data and mathematical models. Professional judgment should always be applied in grain storage decisions.