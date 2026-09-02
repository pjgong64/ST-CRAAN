// ====================================================================
// Rainfall Anomaly / Deficit (Time-Series Sequence)
// Target: Monthly (5-12) 2022
// Historical Baseline: 10 Years (2012 - 2021)
// Dataset: CHIRPS Daily Precipitation (UCSB-CHG/CHIRPS/DAILY)
// ====================================================================

var tambons = ee.FeatureCollection("projects/ee-gongbaac/assets/tha_admin3");

// 1. Define time parameters
//Collect data year by year to avoid running out of memory.
var targetYear = 2022;
var histStartYear = 2012;
var histEndYear = 2021;
// Define the list of target months for the sequence (8=Aug, 9=Sep, 10=Oct, 11=Nov)
var targetMonths = ee.List([5, 6, 7, 8, 9, 10, 11, 12]); 

// ==========================
// 2. Prepare Dataset
// ==========================
// Load CHIRPS Precipitation Dataset
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').select('precipitation');

// CHIRPS resolution is roughly 5.5km (0.05 degrees). We set this scale for accurate reductions.
var chirpsScale = 5566; 

// ==========================
// 3. Core Processing Function (Iterate over Months)
// ==========================
// This function calculates precipitation anomalies and Zonal Statistics for each specified month
var calculateMonthlyDeficit = function(month) {
  month = ee.Number(month);
  var monthStr = ee.String('M').cat(month.format('%02d')); // e.g., M08, M09
  
  // -- A. Calculate Actual Rainfall for the target month & year
  // Filter for the specific month and SUM the daily precipitation
  var rain_actual = chirps.filter(ee.Filter.calendarRange(targetYear, targetYear, 'year'))
                          .filter(ee.Filter.calendarRange(month, month, 'month'))
                          .sum()
                          .rename('Rain_actual');
                          
  // -- B. Calculate Expected Rainfall (Historical Baseline) for the SAME month
  var histYears = ee.List.sequence(histStartYear, histEndYear);
  var histTotals = ee.ImageCollection.fromImages(histYears.map(function(y) {
    // Sum the daily rainfall for this specific month in the historical year 'y'
    return chirps.filter(ee.Filter.calendarRange(y, y, 'year'))
                 .filter(ee.Filter.calendarRange(month, month, 'month'))
                 .sum()
                 .set('year', y);
  }));
  
  // The "Expected" rainfall is the 10-year average of those monthly totals
  var rain_expected = histTotals.mean().rename('Rain_expected');

  // -- C. Calculate Rainfall Deficit / Anomaly
  // Formula: Rain_actual - Rain_expected
  // Note: Negative values = Drought/Deficit. Positive values = Excess rain/Flood risk.
  var rainfall_deficit = rain_actual.subtract(rain_expected).rename('Rainfall_deficit');

  // Combine images to prepare for statistical extraction
  var combinedImage = ee.Image([rain_actual, rain_expected, rainfall_deficit]);

  // -- D. ZONAL STATISTICS for this specific month
  var stats = combinedImage.reduceRegions({
    collection: tambons,
    reducer: ee.Reducer.mean(), 
    scale: chirpsScale, 
    crs: 'EPSG:4326'
  });

  // Remove Geometry and append tracking columns for LSTM sequence building
  return stats.map(function(f) {
    return ee.Feature(null, {
      'adm3_pcode': f.get('adm3_pcode'),
      'target_year': targetYear,
      'month_idx': month,          // Numeric month (8, 9, 10, 11)
      'month_label': monthStr,     // Formatted label (M08, M09...)
      'Rain_actual_mm': f.get('Rain_actual'),      // Actual recorded precipitation
      'Rain_expected_mm': f.get('Rain_expected'),  // 10-year historical baseline
      'Rainfall_deficit_mm': f.get('Rainfall_deficit') // Final calculated anomaly
    });
  });
};

// ==========================
// 4. Execution & Flattening
// ==========================
// Map the processing function over the target months
var monthlyResultsList = targetMonths.map(calculateMonthlyDeficit);

// Flatten the List of Collections into a single continuous DataFrame (Long format)
var finalExportTable = ee.FeatureCollection(monthlyResultsList).flatten();


// ==========================
// 5. Clean up & Export
// ==========================
print("Preview Monthly Rainfall Deficit Results:", finalExportTable.limit(20));

Export.table.toDrive({
  collection: finalExportTable,
  description: 'Thailand_Tambon_Rainfall_Deficit_MonthlySeq' + targetYear,
  fileFormat: 'CSV',
  // Arrange columns logically for downstream Machine Learning preprocessing
  selectors: ['adm3_pcode', 'target_year', 'month_idx', 'month_label', 'Rain_actual_mm', 'Rain_expected_mm', 'Rainfall_deficit_mm']
});
